#ifndef AirfoilSMD_H
#define AirfoilSMD_H

#include "yaml-cpp/yaml.h"

#include "mesh_motion/SMD.h"
#include "vs/vector.h"
#include "vs/tensor.h"

namespace sierra {
namespace nalu {


/** Spring-Mass-Damper system to predict structural response of an airfoil

   3-DOF system with flap, edge and twist


*/
class AirfoilSMD: public SMD
{
public:

    AirfoilSMD(const YAML::Node& node);

    virtual ~AirfoilSMD() {}

    void predict_states();

    void update_timestep(vs::Vector F_np1, vs::Vector M_np1);

    void advance_timestep();

private:

    AirfoilSMD(const AirfoilSMD&) = delete;

    double alpha_;

    vs::Tensor M_;
    vs::Tensor C_;
    vs::Tensor K_;
    
    vs::Vector x_np1_;
    vs::Vector x_n_;
    vs::Vector x_nm1_;

    vs::Vector xdot_np1_;    
    vs::Vector xdot_n_;    
    vs::Vector xdot_nm1_;
    
    vs::Vector v_np1_;
    vs::Vector v_n_;
    vs::Vector v_nm1_;

    vs::Vector a_np1_;
    vs::Vector a_n_;
    vs::Vector a_nm1_;

    vs::Vector f_np1_;
    vs::Vector f_n_;
    vs::Vector f_nm1_;

};

} // namespace nalu
} // namespace sierra

#endif /* AirfoilSMD_H */