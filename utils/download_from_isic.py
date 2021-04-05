import os
import click
import requests

base_url = 'https://isic-archive.com/api/v1'


@click.command()
@click.option('--limit', default=1000, type=click.INT)
@click.option('--offset', default=1000, type=click.INT)
@click.option('--dir', default='gallery', type=click.STRING)
def main(limit, offset, dir):
    image_list = requests.get(f'{base_url}/image?limit={limit}&sort=name&sortdir=1&detail=true&offset={offset}').json()
    for image in image_list:
        try:
            diagnosis = image['meta']['clinical']['diagnosis']
            if not os.path.exists(f'{dir}/{diagnosis}'):
                os.makedirs(f'{dir}/{diagnosis}')

            download_request = requests.get(f"{base_url}/image/{image['_id']}/download")
            download_request.raise_for_status()
            image_path = os.path.join(f'{dir}/{diagnosis}', f"{image['name']}.jpg")
            with open(image_path, 'wb') as output_stream:
                for chunk in download_request:
                    output_stream.write(chunk)
        except:
            pass


if __name__ == '__main__':
    main()
