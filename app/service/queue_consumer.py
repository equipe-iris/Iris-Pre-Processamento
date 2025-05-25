import pika
import json
import boto3
import os
import requests
import tempfile
import threading
from io import BytesIO

from app.service.data_extraction import data_extraction
from app.service.preprocessing_classification import preprocessing_classification
from app.service.classification_predict import classification_predict
from app.service.summarization_predict import summarization_predict
from config import IA_CLASSIFY_RESULTS_URL

def connect_rabbitmq():
    connection = None
    attempts = 0
    max_attempts = 10
    
    while not connection and attempts < max_attempts:
        try:
            attempts += 1
            print(f"Connecting to RabbitMQ (attempt {attempts}/{max_attempts})...")
            connection = pika.BlockingConnection(
                pika.URLParameters(os.getenv('IA_RABBITMQ_URL'))
            )
            print("Connected to RabbitMQ")
        except Exception as e:
            print(f"Connection failed: {e}")
            if attempts >= max_attempts:
                raise Exception("Failed to connect to RabbitMQ")
            import time
            time.sleep(3)
    
    return connection

def setup_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv('IA_S3_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('IA_AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('IA_AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('IA_AWS_REGION')
    )

def process_file_from_s3(file_id, bucket, original_name, s3_client):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=file_id)
        file_content = response['Body'].read()
        
        file_obj = BytesIO(file_content)
        file_obj.name = original_name
        
        df = data_extraction(file_obj, original_name)
        records = df.to_dict(orient='records')

        processed_tickets = []
        for ticket in records:
            tokens = preprocessing_classification(ticket['Interacao'])
            result_classification = classification_predict(tokens, ticket)
            result_summarization = summarization_predict(ticket['Interacao'])
            payload = {
                'id': str(ticket['Id']),
                'title': ticket['Titulo'],
                'in_charge': ticket['Responsavel'],
                'content': ticket['Interacao'],
                'sentiment_rating': result_classification['sentiment_rating'],
                'service_rating': result_classification['service_rating'],
                'summary': result_summarization,
                'start_date': ticket['DataInicio'],
                'end_date': ticket['DataFinal']
            }
            processed_tickets.append(payload)

        result = {
            'file_id': file_id,
            'processed_tickets': processed_tickets
        }
        
        response = requests.post(IA_CLASSIFY_RESULTS_URL, json=[result])
        if response.status_code != 200:
            print(f"Failed to send results: {response.status_code} - {response.text}")
        else:
            print(f"Successfully processed file {original_name} with ID {file_id}")
        
        s3_client.delete_object(Bucket=bucket, Key=file_id)
        print(f"Deleted file {file_id} from S3")
        
    except Exception as e:
        print(f"Error processing file {file_id}: {e}")
        raise

def callback(ch, method, properties, body):
    try:
        message = json.loads(body.decode())
        file_id = message['fileId']
        bucket = message['bucket']
        original_name = message['originalName']
        
        print(f"Processing file {original_name} with ID {file_id}")
        
        s3_client = setup_s3_client()
        process_file_from_s3(file_id, bucket, original_name, s3_client)
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        print(f"Error in callback: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def start_consumer():
    try:
        connection = connect_rabbitmq()
        channel = connection.channel()
        
        queue_name = os.getenv('IA_QUEUE_NAME')
        channel.queue_declare(queue=queue_name, durable=True)
        channel.basic_qos(prefetch_count=1)
        
        print(f"Waiting for messages in {queue_name}")
        channel.basic_consume(queue=queue_name, on_message_callback=callback)
        channel.start_consuming()
        
    except Exception as e:
        print(f"Consumer error: {e}")

def run_consumer_thread():
    consumer_thread = threading.Thread(target=start_consumer, daemon=True)
    consumer_thread.start()
    print("Queue consumer started in background")
