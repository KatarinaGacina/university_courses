import time
import torch.optim
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleMetricEmbedding, IdentityModel
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./mnist/"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_train2 = MNISTMetricDataset(mnist_download_root, split='train', remove_class=0)

    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    print(f"> Loaded {len(ds_train2)} training images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    train_loader2 = DataLoader(
        ds_train2,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 32
    
    model = SimpleMetricEmbedding(1, emb_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    identity_model = IdentityModel().to(device)
    
    identity_representations = compute_representations(identity_model, train_loader, num_classes, 784, device)
    identity_accuracy = evaluate(identity_model, identity_representations, test_loader, device)
    print(f"Test Identity Accuracy: {identity_accuracy * 100:.2f}%")

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.time_ns()

        #train_loss = train(model, optimizer, train_loader, epoch, device)
        train_loss = train(model, optimizer, train_loader2, epoch, device)

        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)
        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            acc1 = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")

        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")
    
    torch.save(model.state_dict(), f'model.pth')
