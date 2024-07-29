#ifndef JSK_RVIZ_PLUGINS_TABLET_VIEW_CONTROLLER_H_
#define JSK_RVIZ_PLUGINS_TABLET_VIEW_CONTROLLER_H_

#ifndef Q_MOC_RUN
#include <rviz/default_plugin/view_controllers/orbit_view_controller.h>
#include <rviz/properties/float_property.h>
#include <rviz/properties/int_property.h>
#include <rviz/properties/enum_property.h>
#include <rviz/properties/vector_property.h>
#include <rviz/properties/bool_property.h>
#include <rviz/properties/tf_frame_property.h>
#include <rviz/properties/editable_enum_property.h>
#include <rviz/properties/ros_topic_property.h>

#include <rviz/render_panel.h>
#include <rviz/view_manager.h>
#include <rviz/ogre_helpers/render_widget.h>
#include "rviz/load_resource.h"
#include "rviz/uniform_string_stream.h"
#include "rviz/display_context.h"
#include "rviz/viewport_mouse_event.h"
#include "rviz/frame_manager.h"
#include "rviz/geometry.h"
#include "rviz/ogre_helpers/shape.h"

#include <ros/subscriber.h>
#include <ros/ros.h>

#include <OGRE/OgreVector3.h>
#include <OGRE/OgreQuaternion.h>
#include <OGRE/OgreViewport.h>
#include <OGRE/OgreQuaternion.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreCamera.h>
#include <OGRE/OgreRenderWindow.h>
#include <RenderSystems/GL/OgreGLDepthBuffer.h>

#include <jsk_rviz_plugins/overlay_image_display.h>
#include <jsk_rviz_plugins/overlay_utils.h>
#include <jsk_rviz_plugins/image_transport_hints_property.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float32.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>

#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <nerf_teleoperation_msgs/NerfRenderRequestAction.h>

#include <random>
#include <math.h>

#endif

namespace rviz
{
  class SceneNode;
  class Shape;
  class BoolProperty;
  class FloatProperty;
  class VectorProperty;
  class QuaternionProperty;
  class TfFrameProperty;
  class EditableEnumProperty;
  class EnumProperty;
  class RosTopicProperty;
  class IntProperty;
  class StringProperty;
}

namespace jsk_rviz_plugins
{
  class OverlayImageDisplay;
}

namespace nerf_view_controller
{

  /** @brief An un-constrained "flying" camera, specified by an eye point, focus point, and up vector. */
  class NerfViewController : public rviz::OrbitViewController
  {
    Q_OBJECT
  public:
    actionlib::SimpleActionClient<nerf_teleoperation_msgs::NerfRenderRequestAction> action_client_;

    NerfViewController();
    // virtual ~NerfViewController();

    /** @brief Do subclass-specific initialization.  Called by
     * ViewController::initialize after context_ and camera_ are set.
     *
     * This version sets up the attached_scene_node, focus shape, and subscribers. */
    virtual void onInitialize();

    /** @brief called by activate().
     *
     * This version calls updateAttachedSceneNode(). */
    // virtual void onActivate();

    virtual void handleMouseEvent(rviz::ViewportMouseEvent &evt);

    /** @brief Configure the settings of this view controller to give,
     * as much as possible, a similar view as that given by the
     * @a source_view.
     *
     * @a source_view must return a valid @c Ogre::Camera* from getCamera(). */

    // /** @brief Called by ViewManager when this ViewController is being made current.
    //  * @param previous_view is the previous "current" view, and will not be NULL.
    //  *
    //  * This gives ViewController subclasses an opportunity to implement
    //  * a smooth transition from a previous viewpoint to the new
    //  * viewpoint.
    //  */
    // virtual void transitionFrom(ViewController *previous_view);

  protected Q_SLOTS:

    /** @brief Called when the render mode property is changed */
    virtual void onRenderModePropertyChanged();

    /** @brief Called when the depth mode property is changed */
    virtual void onDepthModePropertyChanged();

  protected: // methods
    /** @brief Called at 30Hz by ViewManager::update() while this view
     * is active. Override with code that needs to run repeatedly. */
    virtual void update(float dt, float ros_dt);

    void updateRender(float resolution, const cv_bridge::CvImagePtr rgb, const cv_bridge::CvImagePtr depth);

    void resultCb(const actionlib::SimpleClientGoalState &state,
                  const nerf_teleoperation_msgs::NerfRenderRequestResultConstPtr &result);

    void feedbackCb(const nerf_teleoperation_msgs::NerfRenderRequestFeedbackConstPtr &feedback);

    void activeCb();

    void publishCurrentPlacement();

    void epochCallback(const std_msgs::UInt16::ConstPtr &msg);
    void lossCallback(const std_msgs::Float32::ConstPtr &msg);

    // protected Q_SLOTS:

    // void updateTopic();

  protected: // members
    ros::NodeHandle nh_;
    rviz::RosTopicProperty *update_topic_property_;
    jsk_rviz_plugins::ImageTransportHintsProperty *transport_hint_property_;
    jsk_rviz_plugins::OverlayObject::Ptr overlay_;
    ros::Subscriber epoch_sub_;
    ros::Subscriber loss_sub_;

#if ROS_VERSION_MINIMUM(1, 12, 0)
    std::shared_ptr<image_transport::ImageTransport> it_;
#else
    boost::shared_ptr<image_transport::ImageTransport> it_;
#endif

    boost::mutex mat_mutex_;

    bool require_update_;
    cv::Mat mat_;
    cv::Mat overlay_mat_;
    cv::Mat depth_mat_;
    int width_;
    int height_;
    int screen_width_;
    int screen_height_;
    uint8_t render_mode_;
    uint8_t depth_mode_;

    std::string model_name_;
    int32_t client_id_;

    rviz::StringProperty *version_property_;

    rviz::FloatProperty *alpha_property_;
    rviz::FloatProperty *box_size_;
    rviz::FloatProperty *resolution_scale_;
    rviz::FloatProperty *current_resolution_;

    rviz::FloatProperty *training_loss_property_;
    rviz::IntProperty *epoch_property_;
    rviz::StringProperty *model_name_property_;
    rviz::EnumProperty *render_mode_property_;
    rviz::EnumProperty *depth_mode_property_;

    image_transport::Publisher depth_publisher_;

    bool dragging_;
  };

}

#endif