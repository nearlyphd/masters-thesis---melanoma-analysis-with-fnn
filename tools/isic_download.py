import os
import requests
import pandas as pd

BASE_URL = 'https://api.isic-archive.com/api/v2'
PORTION_SIZE = 1000


def flatten_dict(d, sep='.'):
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


def main():
    portion = requests.get(f'{BASE_URL}/images?limit={PORTION_SIZE}&offset=0').json()
    rows = []

    while portion.get('next'):
        for r in portion.get('results'):
            rows.append(pd.DataFrame(flatten_dict(r), index=[0]))

            download_request = requests.get(f"{r.get('files').get('full').get('url')}")
            download_request.raise_for_status()
            image_path = os.path.join(f'images', f"{r['isic_id']}.jpg")
            with open(image_path, 'wb') as output_stream:
                for chunk in download_request:
                    output_stream.write(chunk)

        portion = requests.get(portion.get('next')).json()

    isic_table = pd.concat(rows)
    isic_table.to_csv('isic_table.csv', index=False)


if __name__ == '__main__':
    main()
