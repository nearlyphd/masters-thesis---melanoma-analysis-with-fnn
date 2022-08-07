import click
import pandas as pd

BASE_URL = 'https://api.isic-archive.com/api/v2'
INTERNAL_LIMIT = 1000


def flatten_dict(d, sep='.'):
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


@click.command()
@click.option('--limit', default=1000, type=click.INT)
@click.option('--offset', default=1000, type=click.INT)
@click.option('--csv', default='data.csv', type=click.STRING)
@click.option('--directory', default='isic', type=click.STRING)
def main(limit, offset, csv, directory):
    pass
