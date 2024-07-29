#include <pluginlib/class_list_macros.h>

#include "nerf_view_controller/nerf_cam.hpp"

namespace nerf_view_controller
{
  // Define the action client used to interact with the render server
  typedef actionlib::SimpleActionClient<nerf_teleoperation_msgs::NerfRenderRequestAction> RenderActionClient;

  // Strings for render modes
  static const std::string RENDER_MODE_DYNAMIC = "On Move";
  static const std::string RENDER_MODE_FOVEATED = "Foveated";
  static const std::string RENDER_MODE_CONTINUOUS = "Continuous";

  // Default string for settings
  static const std::string DEFAULT_MODEL = "Undefined";

  // Strings for depth rendering
  static const std::string DEPTH_MODE_RVIZ = "Overlay";
  static const std::string DEPTH_MODE_CLIP = "Occlude";

  NerfViewController::NerfViewController()
      : nh_(""), dragging_(false), action_client_("render_nerf", true)
  {
    // Display the version of the plugin
    version_property_ = new rviz::StringProperty("Version", nerf_view_controller_VERSION, "Version of the plugin", this);
    version_property_->setReadOnly(true);

    // Used to adjust the alpha of the render
    alpha_property_ = new rviz::FloatProperty("Alpha", 0.95,
                                              "The opacity of the render.",
                                              this);

    // What resolution to render at
    resolution_scale_ = new rviz::FloatProperty("Resolution %", 0.5,
                                                "Horizontal  of the NeRF render, vertical is set to match aspect ratio", this);

    // Readout of the curent render resolution
    current_resolution_ = new rviz::FloatProperty("Current Resolution", 0.0,
                                                  "Currently displayed render resolution from the NeRF render", this);
    current_resolution_->setReadOnly(true);

    // AABB to create about the origin
    box_size_ = new rviz::FloatProperty("Crop AABB Size (0 for no crop)", 0.0,
                                        "Size of the box to render", this);

    // Readout of the current training loss
    training_loss_property_ = new rviz::FloatProperty("Training Loss", 0.0, "Training loss of the NeRF model", this);
    training_loss_property_->setReadOnly(true);

    // Readout of the current epoch
    epoch_property_ = new rviz::IntProperty("Epoch", 0, "Current epoch of the NeRF model", this);
    epoch_property_->setReadOnly(true);

    // Try to get the model name from the parameter server
    std::string key;
    if (nh_.searchParam("model_name", key))
      model_name_ = nh_.param<std::string>(key, DEFAULT_MODEL);
    else
      model_name_ = DEFAULT_MODEL;

    model_name_property_ = new rviz::StringProperty("Model Name", QString::fromStdString(model_name_), "Name of the NeRF model to currently rendering", this);
    model_name_property_->setReadOnly(true);

    // What method to render, namely dynamic where it only renders on move, continuous where it renders every 10 steps, and foveated where it renders at multiple resolutions
    render_mode_property_ = new rviz::EnumProperty("Render Mode", QString::fromStdString(RENDER_MODE_DYNAMIC),
                                                   "Select the render mode [On camera move, Continous every 10 steps, Foveated multi resolution].",
                                                   this, SLOT(onRenderModePropertyChanged()));

    render_mode_property_->addOptionStd(RENDER_MODE_DYNAMIC);
    render_mode_property_->addOptionStd(RENDER_MODE_CONTINUOUS);
    render_mode_property_->addOptionStd(RENDER_MODE_FOVEATED);
    render_mode_property_->setStdString(RENDER_MODE_DYNAMIC);
    render_mode_ = 0;

    // What method to render depth, overlay where it renders the depth on top of the RGB, and clip where it occludes the RGB based on the depth
    depth_mode_property_ = new rviz::EnumProperty("Depth Mode", QString::fromStdString(DEPTH_MODE_RVIZ),
                                                  "Select the depth mode [Display RVIZ on top, clip based on NeRF render].",
                                                  this, SLOT(onDepthModePropertyChanged()));

    depth_mode_property_->addOptionStd(DEPTH_MODE_RVIZ);
    depth_mode_property_->addOptionStd(DEPTH_MODE_CLIP);
    depth_mode_property_->setStdString(DEPTH_MODE_RVIZ);
    depth_mode_ = 0;

    // Random client id TODO: make unique across systems
    client_id_ = rand() % 1000000;
  }

  void NerfViewController::onRenderModePropertyChanged()
  {

    if (render_mode_property_->getStdString() == RENDER_MODE_DYNAMIC)
    {
      render_mode_ = 0;
    }
    else if (render_mode_property_->getStdString() == RENDER_MODE_CONTINUOUS)
    {
      render_mode_ = 1;
    }
    else if (render_mode_property_->getStdString() == RENDER_MODE_FOVEATED)
    {
      render_mode_ = 2;
    }
  }

  void NerfViewController::onDepthModePropertyChanged()
  {
    if (depth_mode_property_->getStdString() == DEPTH_MODE_RVIZ)
    {
      depth_mode_ = 0;
    }
    else if (depth_mode_property_->getStdString() == DEPTH_MODE_CLIP)
    {
      depth_mode_ = 1;
    }
  }

  void NerfViewController::epochCallback(const std_msgs::UInt16::ConstPtr &msg)
  {
    epoch_property_->setInt(msg->data);
  }

  void NerfViewController::lossCallback(const std_msgs::Float32::ConstPtr &msg)
  {
    training_loss_property_->setFloat(msg->data);

    // We also look up a change in the model name as this would likely only occur when loss values are sent
    std::string key;
    if (nh_.getParamCached("model_name", model_name_))
      model_name_property_->setStdString(model_name_);
  }

  void NerfViewController::publishCurrentPlacement()
  {
    int width = camera_->getViewport()->getActualWidth();
    int height = camera_->getViewport()->getActualHeight();
    Ogre::Quaternion orientation = camera_->getOrientation();

    nerf_teleoperation_msgs::NerfRenderRequestGoal goal;
    float fl = (height / 2) / tan(camera_->getFOVy().valueDegrees() / 2);
    goal.fov_factor = tan(camera_->getFOVy().valueDegrees() / 2);

    Ogre::Vector3 position = camera_->getPosition();
    goal.pose.position.x = position[0];
    goal.pose.position.y = position[1];
    goal.pose.position.z = position[2];
    goal.pose.orientation.x = orientation.x;
    goal.pose.orientation.y = orientation.y;
    goal.pose.orientation.z = orientation.z;
    goal.pose.orientation.w = orientation.w;
    goal.mode = render_mode_;
    goal.client_id = client_id_;
    goal.frame_id = target_frame_property_->getFrameStd();
    goal.resolution = resolution_scale_->getFloat();

    goal.box_size = box_size_->getFloat();

    goal.width = width;
    goal.height = height;

    action_client_.sendGoal(goal, boost::bind(&NerfViewController::resultCb, this, _1, _2), boost::bind(&NerfViewController::activeCb, this), boost::bind(&NerfViewController::feedbackCb, this, _1));
  }

  void NerfViewController::updateRender(float resolution, const cv_bridge::CvImagePtr rgb_ptr, const cv_bridge::CvImagePtr depth_ptr)
  {
    // Update the printed displayed resolution based on the message
    current_resolution_->setValue(resolution);

    boost::mutex::scoped_lock lock(mat_mutex_);
    ros::Time start_time = ros::Time::now();

    // Copy the images from the message
    rgb_ptr->image.copyTo(mat_);
    depth_ptr->image.copyTo(depth_mat_);

    // 
    width_ = mat_.cols;
    height_ = mat_.rows;

    ros::Duration execution_time = ros::Time::now() - start_time;

    // Update the render at the next chance
    require_update_ = true;
  }

  void NerfViewController::activeCb()
  {
    // do nothing for now...
  }

  void NerfViewController::feedbackCb(const nerf_teleoperation_msgs::NerfRenderRequestFeedbackConstPtr &feedback)
  {
    if (feedback->client_id != client_id_)
      return;

    try
    {
      const cv_bridge::CvImagePtr cv_ptr_rgb = cv_bridge::toCvCopy(feedback->rgb_image, sensor_msgs::image_encodings::BGRA8);
      const cv_bridge::CvImagePtr cv_ptr_depth = cv_bridge::toCvCopy(feedback->depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
      updateRender(feedback->resolution, cv_ptr_rgb, cv_ptr_depth);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  void NerfViewController::resultCb(const actionlib::SimpleClientGoalState &state,
                                    const nerf_teleoperation_msgs::NerfRenderRequestResultConstPtr &result)
  {
    if (result->client_id != client_id_)
      return;

    if (state == actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      try
      {
        const cv_bridge::CvImagePtr cv_ptr_rgb = cv_bridge::toCvCopy(result->rgb_image, sensor_msgs::image_encodings::BGRA8);
        const cv_bridge::CvImagePtr cv_ptr_depth = cv_bridge::toCvCopy(result->depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
        updateRender(result->resolution, cv_ptr_rgb, cv_ptr_depth);
      }
      catch (cv_bridge::Exception &e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
      }
    }
    else
    {
      ROS_ERROR("Action failed: %s", state.toString().c_str());
    }
  }

  void NerfViewController::onInitialize()
  {
    OrbitViewController::onInitialize();

#if ROS_VERSION_MINIMUM(1, 12, 0)
    it_ = std::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(nh_));
#else
    it_ = boost::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(nh_));
#endif

    epoch_sub_ = nh_.subscribe("nerf_step", 1, &NerfViewController::epochCallback, this);
    loss_sub_ = nh_.subscribe("nerf_loss", 1, &NerfViewController::lossCallback, this);

    // Create depth render texture for RViz, TODO: Fix the error where this breaks with rviz image view open....
    Ogre::TexturePtr depth_texture = Ogre::TextureManager::getSingleton().createManual(
        "DepthTexture", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
        Ogre::TEX_TYPE_2D, 640, 480, 0, Ogre::PF_FLOAT32_R, Ogre::TU_RENDERTARGET);

    Ogre::RenderTexture *depth_render_texture = depth_texture->getBuffer()->getRenderTarget();
    Ogre::Viewport *depth_viewport = depth_render_texture->addViewport(camera_);

    depth_viewport->setClearEveryFrame(true);
    depth_viewport->setBackgroundColour(Ogre::ColourValue::Black);
    depth_viewport->setSkiesEnabled(false);
    depth_viewport->setShadowsEnabled(false);
    depth_viewport->setAutoUpdated(false);
    depth_viewport->setCamera(camera_);
    // depth_viewport->setVisibilityMask(0x01);

    screen_width_ = camera_->getViewport()->getActualWidth();
    screen_height_ = camera_->getViewport()->getActualHeight();

    // create image depth publisher
    depth_publisher_ = it_->advertise("nerf_depth", 1);
  }

  void NerfViewController::handleMouseEvent(rviz::ViewportMouseEvent &event)
  {
    OrbitViewController::handleMouseEvent(event);

    if (event.type == QEvent::MouseButtonPress)
      dragging_ = true;
    else if (event.type == QEvent::MouseButtonRelease)
      dragging_ = false;

    // if the scene is moving request new render
    if (event.type == QEvent::MouseButtonPress || event.type == QEvent::MouseButtonRelease || (dragging_ && event.type == QEvent::MouseMove) || event.type == QEvent::Wheel)
    {
      publishCurrentPlacement();
    }
  }

  void NerfViewController::update(float dt, float ros_dt)
  {
    OrbitViewController::update(dt, ros_dt);

    // check if screen size has changed
      if (overlay_ && screen_width_ != camera_->getViewport()->getActualWidth() || screen_height_ != camera_->getViewport()->getActualHeight())
      {
        screen_width_ = camera_->getViewport()->getActualWidth();
        screen_height_ = camera_->getViewport()->getActualHeight();

        // if we have data already loaded, rescale it
        if (!mat_.empty())
          cv::resize(mat_, mat_, cv::Size(screen_width_, screen_height_));

        // request new render for the new screen size/shape      
        publishCurrentPlacement();

        // update the overlay size
        require_update_ = true;
      }


    bool printed = false;

    if (require_update_)
    {
      if (!overlay_)
      {
        static int count = 0;
        rviz::UniformStringStream ss;
        ss << "OverlayImageDisplayObject" << client_id_ << count++;
        
        ROS_DEBUG("Creating overlay object with name %s", ss.str().c_str());

        overlay_.reset(new jsk_rviz_plugins::OverlayObject(ss.str()));
        overlay_->show();
      }

      // Update the overlay size and position
      overlay_->setDimensions(screen_width_, screen_height_);
      overlay_->setPosition(0, 0);
      overlay_->updateTextureSize(screen_width_, screen_height_);

      require_update_ = false;
    }

    if (overlay_ && !mat_.empty() && !depth_mat_.empty())
    {  
      // Update the overlay each frame, this is mainly for realtime depth occlusion from moving robots or other components

      boost::mutex::scoped_lock lock(mat_mutex_);

      cv::Mat depth(screen_height_, screen_width_, CV_32FC1);

      // TODO: GL will sometimes glitch not play well with other image views...
      glReadPixels(0, 0, screen_width_, screen_height_, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data);

      jsk_rviz_plugins::ScopedPixelBuffer buffer = overlay_->getBuffer();
      QImage Hud = buffer.getQImage(*overlay_);

    
      overlay_mat_ = mat_.clone();

      // flip depth image vertically from opengl
      cv::flip(depth, depth, 0);

      // publish depth image from RViz, mainly for testing
      cv_bridge::CvImage cv_image;
      cv_image.image = depth;
      cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
      cv_image.header.stamp = ros::Time::now();
      depth_publisher_.publish(cv_image.toImageMsg());

      // Used to scale the depth values based on OpenGL method
      // https://learnopengl.com/Advanced-OpenGL/Depth-testing
      float near = camera_->getNearClipDistance();
      float far = camera_->getFarClipDistance();

      float alpha = alpha_property_->getFloat();

      // Set the opacity of the overlay image
      for (int i = 0; i < mat_.rows; i++)
      {
        for (int j = 0; j < mat_.cols; j++)
        {
          if (depth_mode_ == 0) // RViz depth overlay
          {
            // In overlay mode just cut out all the objects in the scene
            overlay_mat_.at<cv::Vec4b>(i, j)[3] = depth.at<float>(i, j) < 1.0 || depth_mat_.at<float>(i, j) < 0.0 ? 0 : (int)(255 * alpha);
          }
          else if (depth_mode_ == 1) // clip based on nerf depth
          {
            // For occlude we compare the depth between RViz and NeRF
            float depth_value = depth_mat_.at<float>(i, j);
            depth_value = (1 / depth_value - 1 / near) / (1 / far - 1 / near);

            if (pow(depth.at<float>(i, j), 1) < depth_value || depth_value < 0.0)
              overlay_mat_.at<cv::Vec4b>(i, j)[3] = 0;
            else
              overlay_mat_.at<cv::Vec4b>(i, j)[3] = (int)(255 * alpha);
          }
        }
      }

      // Copy the overlay image to the buffer
      memcpy(Hud.scanLine(0), overlay_mat_.data, overlay_mat_.cols * overlay_mat_.rows * overlay_mat_.elemSize());
      
      glClearDepth(1.0f);
      glClear(GL_DEPTH_BUFFER_BIT);
    }
  }

}

PLUGINLIB_EXPORT_CLASS(nerf_view_controller::NerfViewController, rviz::ViewController)