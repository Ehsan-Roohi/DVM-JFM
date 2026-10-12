import argparse
import matrix
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
matrix.gate(a.index)
