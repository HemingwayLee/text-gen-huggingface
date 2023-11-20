# text-gen-huggingface
* How to run
```
docker build -t myhf .
docker run -it --rm --entrypoint bash myhf
docker run -it --rm -v $(pwd):/app/code/ --entrypoint bash myhf
```

