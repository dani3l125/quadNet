import random
import csv
import os.path as path
import torchvision.models as models

r_min = -10000.0
r_max = 10000.0
header_p = ['a', 'b', 'c']
header_r = ['root1', 'root2']

def main():
    if not path.exists('Data/parameters.csv'):  # generate dataset if needed
        # TODO: add one root, no roots. deal with loss function
        with open('Data/parameters.csv', 'w', encoding='UTF8') as p, open('Data/roots.csv', 'w', encoding='UTF8') as r:
            writer_r = csv.writer(r)
            writer_r.writerow(header_r)
            writer_p = csv.writer(p)
            writer_p.writerow(header_p)
            for i in range(1000000):
                r1 = random.uniform(r_min, r_max)
                r2 = random.uniform(r_min, r_max)
                if r1 > r2:  # assuming ordered roots
                    temp = r1
                    r1 = r2
                    r2 = temp
                a = 1
                b = r1 + r2
                c = r1 * r2
                if random.getrandbits(1): # can remove
                    a = random.uniform(r_min, r_max)
                    b *= a
                    c *= a
                writer_p.writerow([r1, r2])
                writer_r.writerow([a, b, c])

    #TODO: visualize with pandas

    # Create network
    net = models.resnet18()  #TODO find best model


if __name__ == "__main__":
    main()

