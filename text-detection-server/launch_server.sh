# !docker run -d --rm  --name="ovms" -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest \
# --model_path /models/detection/ --model_name detection --port 9000

docker run -d --rm  --name="ovms" -v $(pwd)/models:/models \
                -p 9000:9000 openvino/model_server:latest  \
                --model_path /models/text-detection/       \
                --model_name text-detection --port 9000