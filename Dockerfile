FROM python:3.8-slim
COPY . /ProductReviw
WORKDIR /ProductReviw
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
ENV PORT 8000
ENV HOST 0.0.0.0
CMD [ "sh", "-c", "streamlit run --server.port ${PORT} --server.address ${HOST} app1.py" ]
#CMD streamlit run app1.py