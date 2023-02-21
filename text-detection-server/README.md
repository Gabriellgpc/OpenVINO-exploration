The models need to be placed and mounted in a particular directory structure and according to the following rules:

```
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── ir_model.bin
│       └── ir_model.xml
├── model2
│   └── 1
│       ├── ir_model.bin
│       ├── ir_model.xml
│       └── mapping_config.json
├── model3
│    └── 1
│        └── model.onnx
├── model4
│      └── 1
│        ├── model.pdiparams
│        └── model.pdmodel
└── model5
       └── 1
         └── TF_fronzen_model.pb
```


* Each model should be stored in a dedicated directory, e.g. model1 and model2.

* Each model directory should include a sub-folder for each of its versions (1,2, etc). The versions and their folder names should be positive integer values.

* Note: In execution, the versions are enabled according to a pre-defined version policy. If the client does not specify the version number in parameters, by default, the latest version is served.

* Every version folder must include model files, that is, `.bin` and `.xml` for OpenVINO IR, `.onnx` for ONNX, `.pdiparams` and `.pdmodel` for Paddle Paddle, and `.pb` for TensorFlow. The file name can be arbitrary.


The required Model Server parameters are listed below. For additional configuration options, see the [Model Server Parameters section](https://docs.openvino.ai/latest/ovms_docs_parameters.html#doxid-ovms-docs-parameters).

<table class="table">
<colgroup>
<col style="width: 20%" />
<col style="width: 80%" />
</colgroup>
<tbody>
<tr class="row-odd"><td><p><cite>–rm</cite></p></td>
<td><div class="line-block">
<div class="line">remove the container when exiting the Docker container</div>
</div>
</td>
</tr>
<tr class="row-even"><td><p><cite>-d</cite></p></td>
<td><div class="line-block">
<div class="line">runs the container in the background</div>
</div>
</td>
</tr>
<tr class="row-odd"><td><p><cite>-v</cite></p></td>
<td><div class="line-block">
<div class="line">defines how to mount the model folder in the Docker container</div>
</div>
</td>
</tr>
<tr class="row-even"><td><p><cite>-p</cite></p></td>
<td><div class="line-block">
<div class="line">exposes the model serving port outside the Docker container</div>
</div>
</td>
</tr>
<tr class="row-odd"><td><p><cite>openvino/model_server:latest</cite></p></td>
<td><div class="line-block">
<div class="line">represents the image name; the ovms binary is the Docker entry point</div>
<div class="line">varies by tag and build process - see tags: <a class="reference external" href="https://hub.docker.com/r/openvino/model_server/tags/">https://hub.docker.com/r/openvino/model_server/tags/</a> for a full tag list.</div>
</div>
</td>
</tr>
<tr class="row-even"><td><p><cite>–model_path</cite></p></td>
<td><div class="line-block">
<div class="line">model location, which can be:</div>
<div class="line">a Docker container path that is mounted during start-up</div>
<div class="line">a Google Cloud Storage path <cite>gs://&lt;bucket&gt;/&lt;model_path&gt;</cite></div>
<div class="line">an AWS S3 path <cite>s3://&lt;bucket&gt;/&lt;model_path&gt;</cite></div>
<div class="line">an Azure blob path <cite>az://&lt;container&gt;/&lt;model_path&gt;</cite></div>
</div>
</td>
</tr>
<tr class="row-odd"><td><p><cite>–model_name</cite></p></td>
<td><div class="line-block">
<div class="line">the name of the model in the model_path</div>
</div>
</td>
</tr>
<tr class="row-even"><td><p><cite>–port</cite></p></td>
<td><div class="line-block">
<div class="line">the gRPC server port</div>
</div>
</td>
</tr>
<tr class="row-odd"><td><p><cite>–rest_port</cite></p></td>
<td><div class="line-block">
<div class="line">the REST server port</div>
</div>
</td>
</tr>
</tbody>
</table>

If the serving port ```9000``` is already in use, please switch it to another avaiable port on your system.