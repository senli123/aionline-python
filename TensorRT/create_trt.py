def build(self, model_path, gpuID):
        def build_engine():
            with trt.Builder(self.TRT_LOGGER) as builder, 
            builder.create_network(common.EXPLICIT_BATCH) as network, 
            trt.OnnxParser(network,self.TRT_LOGGER) as parser:
                builder.max_workspace_size = 1 << 28 # 256MiB
                builder.max_batch_size = self.Tparams.BatchSize
                # Parse model file
                onnx_file_path = os.path.join(model_path,self.Tparams.OnnxFileName)
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found, please first to generate it.'.format(self.Tparams.OnnxFileName))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(self.Tparams.OnnxFileName))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    if not parser.parse(model.read()):
                        print('ERROR: Failed to parse the ONNX file.')
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None
                network.get_input(0).shape = self.Tparams.shape
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(self.Tparams.OnnxFileName))
                engine = builder.build_cuda_engine(network)
                print('Completed creating Engine')
                with open(engine_file_path, 'wb') as f:
                    f.write(engine.serialize())
                return engine
        #engine_file_path = os.path.join(model_path,self.Tparams.TRTFileName)
        engine_file_path =- model_path
        if os.path.exists(engine_file_path):
            print('Reading engine from file {}'.format(engine_file_path))
            with open(engine_file_path, 'rb') as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()