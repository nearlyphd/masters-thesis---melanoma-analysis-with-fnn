import os
import click
import random
import shutil


@click.command()
@click.option('--source', default=None, type=click.STRING)
@click.option('--destination', default=None, type=click.STRING)
@click.option('--size', default=None, type=click.FLOAT)
def main(source, destination, size):
    for im_class in os.listdir(source):
        if not os.path.exists(f'{destination}/{im_class}'):
            os.makedirs(f'{destination}/{im_class}')
        images = os.listdir(f'{source}/{im_class}')
        images_to_move = random.sample(images, k=round(len(images) * size))
        for im in images_to_move:
            shutil.move(f'{source}/{im_class}/{im}', f'{destination}/{im_class}/{im}')


if __name__ == '__main__':
    main()
