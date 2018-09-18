# import os
# import json
# import unittest

# from rastervision.rv_config import RVConfig

# class TestRVConfig(unittest.TestCase):

#     def test_config(self):
#         rv_conf = RVConfig.get_instance()
#         print(rv_conf.config('batch_job_queue', namespace='AWS_BATCH'))

#         n = rv_conf.get_subconfig("nonexist")
#         print(n('files', default=''))

#         plugin_files = json.loads(rv_conf.config('files', namespace='PLUGINS'))

#         print(plugin_files)

#         from pluginbase import PluginBase
#         plugin_base = PluginBase(package='tests.plugins')

#         plugin_source = plugin_base.make_plugin_source(
#             searchpath=plugin_files)

#         for plugin_name in plugin_source.list_plugins():
#             plugin = plugin_source.load_plugin(plugin_name)
#             print("{}: {}".format(plugin_name, plugin.what(10)))

# if __name__ == '__main__':
#     unittest.main()
