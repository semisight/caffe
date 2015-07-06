#ifndef CAFFE_PLUGIN_HPP_
#define CAFFE_PLUGIN_HPP_

#include <cstdlib>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief An API for the plugins to implement.
 *
 * This struct is returned by the GetAPI() function in the plugin .so. The
 * Creator functions GetFloatLayer and GetDoubleLayer are registered in the
 * LayerRegistry and return new smart refs to a subclass of Layer.
 */
struct API {
  const char* (*GetName)();
  shared_ptr<Layer<float> > (*GetFloatLayer)(const LayerParameter &);
  shared_ptr<Layer<double> > (*GetDoubleLayer)(const LayerParameter &);

  API(const char* (*Name)(),
    shared_ptr<Layer<float> > (*FloatLayer)(const LayerParameter &),
    shared_ptr<Layer<double> > (*DoubleLayer)(const LayerParameter &));
};

// This is a big ugly macro that does the repetitive API stuff. Put in cpp file
#define INSTANTIATE_PLUGIN(stringname, classname) \
  const char* GetName() { return #stringname; } \
  template <typename Dtype> \
  shared_ptr<Layer<Dtype> > GetLayer(const LayerParameter &param) { \
    return shared_ptr<Layer<Dtype> >(new classname<Dtype>(param)); \
  } \
  extern "C" API* GetAPI() { \
    static API api(GetName, GetLayer<float>, GetLayer<double>); \
    return &api; \
  }

void LoadPlugins(const char* path);
void LoadPlugin(const char* path);

// Class to load plugins from a directory.
class PluginPathLoader {
  public:
    PluginPathLoader(void) {
      char* envvar = std::getenv("CAFFE_PLUGIN_PATH");
      if (envvar == NULL) {
        LOG(WARNING) << "Skipping plugin loading";
        return;
      }
      vector<string> paths;
      boost::split(paths, envvar, boost::is_any_of(":"));
      for (vector<string>::iterator it = paths.begin(); it != paths.end(); ++it) {
        LOG(INFO) << "Loading plugins from " << *it;
        LoadPlugins(it->c_str());
      }
    }
};

#define LOAD_PLUGINS() \
  static PluginPathLoader g_loader

}

#endif
