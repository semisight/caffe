#include <dlfcn.h>

#include "caffe/plugin.hpp"

namespace fs = boost::filesystem;

namespace caffe {

API::API(const char* (*Name)(),
  shared_ptr<Layer<float> > (*FloatLayer)(const LayerParameter &),
  shared_ptr<Layer<double> > (*DoubleLayer)(const LayerParameter &)) {
  this->GetName = Name;
  this->GetFloatLayer = FloatLayer;
  this->GetDoubleLayer = DoubleLayer;
}

void LoadPlugins(const char* path) {
  fs::path dir_path(path);
  if (!fs::is_directory(dir_path)) {
    LOG(WARNING) << dir_path.string() << " does not exist, skipping";
    return;
  }
  for (fs::directory_iterator it(dir_path); it != fs::directory_iterator(); ++it) {
    fs::path plugin_path = it->path();
    if (plugin_path.extension() != ".so") {
      LOG(WARNING) << "Skipping file " << plugin_path;
      continue;
    }

    fs::path abs_path = fs::absolute(plugin_path);
    LoadPlugin(abs_path.c_str());
  }
}

typedef API* (*api_fn)();

void LoadPlugin(const char* path) {
  char* error;
  LOG(INFO) << "Loading " << path;
  void* plugin = dlopen(path, RTLD_NOW);
  if (plugin == NULL)
    LOG(FATAL) << "Could not load plugin: " << dlerror();
  void* get_api = dlsym(plugin, "GetAPI");
  if (get_api == NULL && (error = dlerror()) != NULL)
    LOG(FATAL) << error;
  API* api = ((api_fn)(get_api))();
  LayerRegistry<float>::AddCreator(api->GetName(), api->GetFloatLayer);
  LayerRegistry<double>::AddCreator(api->GetName(), api->GetDoubleLayer);
  LOG(INFO) << "Successfully added plugin layer " << api->GetName();
}

LOAD_PLUGINS();

}
