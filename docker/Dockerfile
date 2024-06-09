FROM python:3.11-slim

WORKDIR /optics

RUN pip install jupyterlab

EXPOSE 8888

COPY ./requirements.txt .  

# Update dependences for cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]




