import os
import click
import requests

base_url = 'https://isic-archive.com/api/v1'


@click.command()
@click.option('--size', default=1000, type=click.INT)
def main(size):
    image_list = requests.get(f'{base_url}/image?limit={size}&sort=name&sortdir=1&detail=true').json()
    for index, image in enumerate(image_list):
        try:
            benign_malignant = image['meta']['clinical']['benign_malignant']
            if not os.path.exists(f'gallery/{benign_malignant}'):
                os.makedirs(f'gallery/{benign_malignant}')

            print(image['_id'])
            download_request = requests.get(f"{base_url}/image/{image['_id']}/download")
            download_request.raise_for_status()
            image_path = os.path.join(f'gallery/{benign_malignant}', f"{image['name']}.jpg")
            with open(image_path, 'wb') as output_stream:
                for chunk in download_request:
                    output_stream.write(chunk)
        except:
            print(f'connection issue')


if __name__ == '__main__':
    main()
