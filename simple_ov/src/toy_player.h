#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "feat/FeatureInitializerOptions.h"
#include "core/VioManagerOptions.h"

namespace ov_core {
struct ImuData;
struct CameraData;
class TrackBase;
class FeatureInitializer;
}

namespace ov_init {
class InertialInitializer;
} 

namespace ov_msckf{
    class VioManagerOptions;
    class State;
    class StateHelper;
    class UpdaterMSCKF;
    class UpdaterSLAM;
    class Propagator;
}

namespace simple_ov {

/*
 *@brief: light version of the VIO system manager to handle initialization,
 *state propagate and filtering/update
*/
class ToyManager{
    public:
        ToyManager(ov_msckf::VioManagerOptions &options_);

        void feed_imu_data(const ov_core::ImuData &imu_msg);

        void feed_camera_data(const ov_core::CameraData &cam_msg);

    protected:
        void track_image_and_update(const ov_core::CameraData &cam_msg);

        void do_feature_propagate_update(const ov_core::CameraData &cam_msg);

        ov_msckf::VioManagerOptions &options;

        std::shared_ptr<ov_msckf::State> state;

        std::shared_ptr<ov_msckf::Propagator> propagator;

        /// Our sparse feature tracker (klt or descriptor)
        std::shared_ptr<ov_core::TrackBase> trackFEATS;

        /// Our MSCKF feature updater
        std::shared_ptr<ov_msckf::UpdaterMSCKF> updaterMSCKF;

        /// Our SLAM/ARUCO feature updater
        std::shared_ptr<ov_msckf::UpdaterSLAM> updaterSLAM;

}
}  // namespace simple_ov
