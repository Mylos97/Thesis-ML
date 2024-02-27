import dataloader
from training import fit

def main():
    X1, X2, Y1, Y2 = dataloader.load_data()
    fit(X1=X1, X2=X2, Y1=Y1, Y2=Y2)

if __name__ == "__main__":
    main()