#include <gtest/gtest.h>
#include <limits>

#include "mesh_motion/MotionDeformingInteriorKernel.h"
#include "mesh_motion/MotionRotationKernel.h"
#include "mesh_motion/MotionScalingKernel.h"
#include "mesh_motion/MotionTranslationKernel.h"

#include "mesh_motion/NgpMotion.h"
#include "mesh_motion/FrameBase.h"
#include "mesh_motion/SMD.h"
#include "mesh_motion/AirfoilSMD.h"
#include "mesh_motion/MotionAirfoilSMDKernel.h"

#include "UnitTestRealm.h"

namespace {

const double testTol = 1e-14;

std::vector<double>
transform(
  const sierra::nalu::mm::TransMatType& transMat,
  const sierra::nalu::mm::ThreeDVecType& xyz)
{
  std::vector<double> transCoord(3, 0.0);

  // perform matrix multiplication between transformation matrix
  // and original coordinates to obtain transformed coordinates
  for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; d++) {
    transCoord[d] = transMat[d * sierra::nalu::mm::matSize + 0] * xyz[0] +
                    transMat[d * sierra::nalu::mm::matSize + 1] * xyz[1] +
                    transMat[d * sierra::nalu::mm::matSize + 2] * xyz[2] +
                    transMat[d * sierra::nalu::mm::matSize + 3];
  }

  return transCoord;
}

} // namespace

TEST(meshMotion, rotation_omega)
{
  // create a yaml node describing rotation
  const std::string rotInfo = "omega: 3.0              \n"
                              "centroid: [0.3,0.5,0.0] \n";

  YAML::Node rotNode = YAML::Load(rotInfo);

  // initialize the mesh rotation class
  sierra::nalu::MotionRotationKernel rotClass(rotNode);

  // build transformation
  const double time = 3.5;
  sierra::nalu::mm::ThreeDVecType xyz{2.5, 1.5, 6.5};
  sierra::nalu::mm::TransMatType transMat =
    rotClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  const double gold_norm_x = 0.133514518380489;
  const double gold_norm_y = -1.910867599933667;
  const double gold_norm_z = 6.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  sierra::nalu::mm::ThreeDVecType tmp;
  sierra::nalu::mm::ThreeDVecType vel =
    rotClass.compute_velocity(time, transMat, tmp, xyz);

  const double gold_norm_vx = -3.0;
  const double gold_norm_vy = 6.6;
  const double gold_norm_vz = 0.0;

  EXPECT_NEAR(vel[0], gold_norm_vx, testTol);
  EXPECT_NEAR(vel[1], gold_norm_vy, testTol);
  EXPECT_NEAR(vel[2], gold_norm_vz, testTol);
}

namespace sierra {
namespace nalu {

TEST(meshMotion, frame_rotation)
{
  const std::string smd_info = "centroid: [0.3, 0.5, 0.0] \n"
                              "mass_matrix: [7000.0, -90.0, 60.0, -90.0, 6400.0, 1800.0, 60.0, 1800.0, 7800] \n"
                              "stiffness_matrix: [131000.0, 7700.0, -97000.0, 7700.0, 66000.0, -31000.0, -97000.0, -31000.0, 4800000.0] \n"
                              "damping_matrix: [300.0, 8.0, -70.0, 8.0, 200.0, 20.0, -70.0, 20.0, 3700.0] \n"
                              "x_init: [0.1, 0.2, 0.0] \n"
                              "xdot_init: [0.1, 0.2, 0.5235987755982988] \n"
                              "xdot_nm1:  [0.1, 0.2, 0.5235987755982988] \n"
                              "loads_scale: 0.75 \n"
                              "alpha : -0.05 \n";

  YAML::Node motion_def = YAML::Load(smd_info);
  
  // initialize an smd_ object for use in the following
  // make the smd object the first of an array/list

  int i = 0;
  int num_motions = 1;

  const double time = 3.5;
  const double dt = 1.0;

  sierra::nalu::mm::ThreeDVecType xyz{2.5+0.3, 1.5+0.5, 6.5};
  sierra::nalu::mm::ThreeDVecType cxyz{0.0, 0.0, 0.0};

  // Below here are lines copied from FrameSMD with minimal edits.

  std::vector<std::unique_ptr<SMD>> smd_;
  smd_.resize(num_motions);
  smd_[i].reset(new AirfoilSMD(motion_def));

  std::vector<std::unique_ptr<NgpMotion>> motionKernels_;
  motionKernels_.resize(num_motions); 
  motionKernels_[i].reset(new MotionAirfoilSMDKernel(motion_def));

  auto ngpKernels = nalu_ngp::create_ngp_view<NgpMotion>(motionKernels_);

  NgpMotion* kernel = ngpKernels(i);


  // Additional lines not in the FrameSMD - get some state values for the np1 state
  smd_[i]->setup(dt);
  smd_[i]->predict_states();

  // Inputs for the motion
  vs::Vector trans_disp = smd_[i]->get_trans_disp();
  const double rot_angle = smd_[i]->get_rot_disp();
  vs::Vector axis = smd_[i]->get_rot_axis();
  vs::Vector origin = smd_[i]->get_origin(); 

  // Generate Transformation
  mm::TransMatType currTransMat = kernel->build_transformation(time, trans_disp, origin, axis, rot_angle);

  // Generate velocity part of unit test
  vs::Vector trans_vel = smd_[i]->get_trans_vel();
  vs::Vector rot_vel = smd_[i]->get_rot_vel();

  std::vector<double> norm = transform(currTransMat, xyz);
  for(int d = 0; d < 3; ++d){
    cxyz[d] = norm[d];
  }

  mm::ThreeDVecType vel =
      kernel->compute_velocity(time, cxyz, origin+trans_disp, trans_vel, rot_vel);

  // Check results of unit test
 
  const double gold_norm_x = 1.6150635094610968 + 0.3;
  const double gold_norm_y = 2.949038105676658 + 0.5;
  const double gold_norm_z = 6.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  const double gold_norm_vx = -1.2346732310857051;
  const double gold_norm_vy = 0.9409255209476621;
  const double gold_norm_vz = 0.0;

  EXPECT_NEAR(vel[0], gold_norm_vx, testTol);
  EXPECT_NEAR(vel[1], gold_norm_vy, testTol);
  EXPECT_NEAR(vel[2], gold_norm_vz, testTol);
}

} // End nalu namespace
} // End seirra namespace

TEST(meshMotion, rotation_angle)
{
  // create a yaml node describing rotation
  const std::string rotInfo = "angle: 180            \n"
                              "centroid: [0.3,0.5,0.0] \n";

  YAML::Node rotNode = YAML::Load(rotInfo);

  // initialize the mesh rotation class
  sierra::nalu::MotionRotationKernel rotClass(rotNode);

  // build transformation
  const double time = 0.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5, 1.5, 6.5};
  sierra::nalu::mm::TransMatType transMat =
    rotClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  const double gold_norm_x = -1.9;
  const double gold_norm_y = -0.5;
  const double gold_norm_z = 6.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}

TEST(meshMotion, scaling)
{
  // create a yaml node describing scaling
  const std::string scaleInfo = "factor: [2.0,2.0,1.0] \n"
                                "centroid: [0.3,0.5,0.0] \n";

  YAML::Node scaleNode = YAML::Load(scaleInfo);

  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();

  // initialize the mesh scaling class
  sierra::nalu::MotionScalingKernel scaleClass(realm.meta_data(), scaleNode);

  // build transformation
  const double time = 0.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5, 1.5, 6.5};
  sierra::nalu::mm::TransMatType transMat =
    scaleClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  const double gold_norm_x = 4.7;
  const double gold_norm_y = 2.5;
  const double gold_norm_z = 6.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}

TEST(meshMotion, translation_velocity)
{
  // create a yaml node describing translation
  const std::string transInfo = "start_time: 15.0        \n"
                                "end_time: 25.0          \n"
                                "velocity: [1.5, 3.5, 2] \n";

  YAML::Node transNode = YAML::Load(transInfo);

  // initialize the mesh translation class
  sierra::nalu::MotionTranslationKernel transClass(transNode);

  // build transformation at t = 10.0
  double time = 10.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5, 1.5, 6.5};
  sierra::nalu::mm::TransMatType transMat =
    transClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  double gold_norm_x = xyz[0];
  double gold_norm_y = xyz[1];
  double gold_norm_z = xyz[2];

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  // build transformation at t = 20.0
  time = 20.0;
  transMat = transClass.build_transformation(time, xyz);
  norm = transform(transMat, xyz);

  gold_norm_x = 10.0;
  gold_norm_y = 19.0;
  gold_norm_z = 16.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);

  // build transformation at t = 30.0
  time = 30.0;
  transMat = transClass.build_transformation(time, xyz);
  norm = transform(transMat, xyz);

  gold_norm_x = 17.5;
  gold_norm_y = 36.5;
  gold_norm_z = 26.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}

TEST(meshMotion, translation_displacement)
{
  // create a yaml node describing translation
  const std::string transInfo = "displacement: [1.5, 3.5, 2] \n";

  YAML::Node transNode = YAML::Load(transInfo);

  // initialize the mesh translation class
  sierra::nalu::MotionTranslationKernel transClass(transNode);

  // build transformation
  const double time = 0.0;
  sierra::nalu::mm::ThreeDVecType xyz{2.5, 1.5, 6.5};
  sierra::nalu::mm::TransMatType transMat =
    transClass.build_transformation(time, xyz);
  std::vector<double> norm = transform(transMat, xyz);

  const double gold_norm_x = 4.0;
  const double gold_norm_y = 5.0;
  const double gold_norm_z = 8.5;

  EXPECT_NEAR(norm[0], gold_norm_x, testTol);
  EXPECT_NEAR(norm[1], gold_norm_y, testTol);
  EXPECT_NEAR(norm[2], gold_norm_z, testTol);
}

TEST(meshMotion, deform_interior_outside_node)
{
  // create a yaml node describing translation
  const std::string deformInfo = "xyz_min: [0,0,0]         \n"
                                 "xyz_max: [15,5,5]        \n"
                                 "amplitude: [1.5,0.0,1.5] \n"
                                 "frequency: [0.1,0.0,0.1] \n"
                                 "centroid: [7.5,2.5,2.5]  \n";

  YAML::Node deformNode = YAML::Load(deformInfo);

  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();

  // initialize the mesh translation class
  sierra::nalu::MotionDeformingInteriorKernel deformClass(
    realm.meta_data(), deformNode);

  // build transformation
  const double time = 1.66666667;
  sierra::nalu::mm::ThreeDVecType xyz{9, 7, 3.5};
  sierra::nalu::mm::TransMatType transMat =
    deformClass.build_transformation(time, xyz);
  std::vector<double> currCoord = transform(transMat, xyz);

  const double gold_norm_x = 9.0;
  const double gold_norm_y = 7.0;
  const double gold_norm_z = 3.5;

  EXPECT_NEAR(currCoord[0], gold_norm_x, testTol);
  EXPECT_NEAR(currCoord[1], gold_norm_y, testTol);
  EXPECT_NEAR(currCoord[2], gold_norm_z, testTol);

  sierra::nalu::mm::ThreeDVecType tmp;
  sierra::nalu::mm::ThreeDVecType vel =
    deformClass.compute_velocity(time, transMat, xyz, tmp);

  const double gold_norm_vx = 0.0;
  const double gold_norm_vy = 0.0;
  const double gold_norm_vz = 0.0;

  EXPECT_NEAR(vel[0], gold_norm_vx, testTol);
  EXPECT_NEAR(vel[1], gold_norm_vy, testTol);
  EXPECT_NEAR(vel[2], gold_norm_vz, testTol);
}

TEST(meshMotion, deform_interior_inside_node)
{
  // create a yaml node describing translation
  const std::string deformInfo = "xyz_min: [0,0,0]         \n"
                                 "xyz_max: [15,5,5]        \n"
                                 "amplitude: [1.5,0.0,1.5] \n"
                                 "frequency: [0.1,0.0,0.1] \n"
                                 "centroid: [7.5,2.5,2.5]  \n";

  YAML::Node deformNode = YAML::Load(deformInfo);

  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();

  // initialize the mesh translation class
  sierra::nalu::MotionDeformingInteriorKernel deformClass(
    realm.meta_data(), deformNode);

  // build transformation
  const double time = 1.66666667;
  sierra::nalu::mm::ThreeDVecType xyz{9.0, 4, 1.5};
  sierra::nalu::mm::TransMatType transMat =
    deformClass.build_transformation(time, xyz);
  std::vector<double> currCoord = transform(transMat, xyz);

  const double gold_norm_x = 9.7500000027207;
  const double gold_norm_y = 4.0;
  const double gold_norm_z = 0.749999997279301;

  EXPECT_NEAR(currCoord[0], gold_norm_x, testTol);
  EXPECT_NEAR(currCoord[1], gold_norm_y, testTol);
  EXPECT_NEAR(currCoord[2], gold_norm_z, testTol);

  sierra::nalu::mm::ThreeDVecType tmp;
  sierra::nalu::mm::ThreeDVecType vel =
    deformClass.compute_velocity(time, transMat, xyz, tmp);

  const double gold_norm_vx = 0.816209714892358;
  const double gold_norm_vy = 0.0;
  const double gold_norm_vz = -0.816209714892358;

  EXPECT_NEAR(vel[0], gold_norm_vx, testTol);
  EXPECT_NEAR(vel[1], gold_norm_vy, testTol);
  EXPECT_NEAR(vel[2], gold_norm_vz, testTol);
}
