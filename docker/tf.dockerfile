FROM tensorflow/tensorflow:nightly-gpu
RUN pip install scipy click
CMD tail -f /dev/null