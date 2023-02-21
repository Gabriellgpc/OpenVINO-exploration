# OpenVINO-exploration

```
$ pip install openvino-dev
```

```
$ omz_downloader --print_all
```

```
$ omz_downloader -o model --name mobilenet-v3-small-1.0-224-tf
```

## Model Converter Usage

```
$ omz_converter -d model --name mobilenet-v3-small-1.0-224-tf
```

# References
* https://github.com/openvinotoolkit

# TODO
- [x] Use cache in face detection exploration
- [ ] Face detection Model server
- [ ] Face similarity analysis
- [ ] Face similarity analysis model server
- [ ] Optimize preprocessing
- [ ] Face detection from TensorFlow or Pytorch
- [ ] Vision super resolution
- [ ] Vision deblur