# Image Classifier


Mobilenet v2 image classification model exposed as a Flask api endpoint in a docker container.


Build docker image:
```
docker build -t image-classifier .
```


Run docker container:
```
docker run -d -p 80:80 image-classifier
```


Use this command to send image to the container for inference:
```
curl -X POST http://127.0.0.1:80/image -d '{"url": "<image url>"}' 
```
