import os
import click
import requests


training_dir = './training_gallery'
testing_dir = './testing_gallery'
base_url = 'https://isic-archive.com/api/v1'


@click.command()
@click.option('--training-size', default=1000, type=click.INT)
@click.option('--testing-size', default=500, type=click.INT)
def main(training_size, testing_size):
    total_size = training_size + testing_size
    image_list = requests.get(f'{base_url}/image?limit={total_size}&sort=name&sortdir=1&detail=true').json()
    for index, image in enumerate(image_list):
        try:
            folder = training_dir if index < training_size else testing_dir
            benign_malignant = image['meta']['clinical']['benign_malignant']
            if not os.path.exists(f'{folder}/{benign_malignant}'):
                os.makedirs(f'{folder}/{benign_malignant}')

            print(image['_id'])
            download_request = requests.get(f"{base_url}/image/{image['_id']}/download")
            download_request.raise_for_status()
            image_path = os.path.join(f'{folder}/{benign_malignant}', f"{image['name']}.jpg")
            with open(image_path, 'wb') as output_stream:
                for chunk in download_request:
                    output_stream.write(chunk)
        except:
            print(f'connection issue')



if __name__ == '__main__':
    main()
