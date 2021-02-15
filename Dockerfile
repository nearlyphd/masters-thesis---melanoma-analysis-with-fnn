FROM tensorflow/tensorflow:nightly-gpu-jupyter
RUN pip install scipy click
EXPOSE 8888
EXPOSE 6006
CMD jupyter notebook --allow-root --ip='0.0.0.0' --notebook-dir='/tf/notebooks'
