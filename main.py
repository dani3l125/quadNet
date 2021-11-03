import random
import csv

r_min = -10000.0
r_max = 10000.0
header_p = ['a', 'b', 'c']
header_r = ['root1', 'root2']

def main():
    print("jere")
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
            if random.getrandbits(1):
                a = random.uniform(r_min, r_max)
                b *= a
                c *= a
            writer_p.writerow([r1, r2])
            writer_r.writerow([a, b, c])

            p = [a, b, c]
            r = [r1, r2]

if __name__ == "__main__":
    main()

