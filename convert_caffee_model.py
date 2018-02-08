import coremltools

coreml_model = coremltools.converters.caffe.convert(('./GazeCapture-master/models/snapshots/itracker_iter_92000.caffemodel', './GazeCapture-master/models/itracker_deploy.prototxt'))


# Now save the model
coremltools.utils.save_spec(coreml_model, 'eye_tracking.mlmodel')
