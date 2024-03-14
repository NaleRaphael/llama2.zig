"""
A script to extract a few tensors from weights to check with mmap data.
DO NOT print extra content to STDIN. Because STDIN is the way we passing
data to test script in zig.
"""
from argparse import ArgumentParser
import struct
import torch
import numpy as np

def write_array(file, array):
    assert array.dtype == np.float32
    b = struct.pack(f"{len(array)}f", *array)
    file.write(b)

def main():
    parser = ArgumentParser()
    parser.add_argument("fn_chkpt", type=str, default="stories15M.pt")
    parser.add_argument("--fn_output", type=str, default="weight_samples.bin")
    args = parser.parse_args()

    checkpoint_dict = torch.load(args.fn_chkpt, map_location="cpu")
    state_dict = checkpoint_dict["model"]

    # currently we only get data from the first layer as it would be the first
    # few elements in mmap data
    tok_embeddings = state_dict["tok_embeddings.weight"]
    rms_att_weight = state_dict["layers.0.attention_norm.weight"]
    wq = state_dict["layers.0.attention.wq.weight"]
    wk = state_dict["layers.0.attention.wk.weight"]
    wv = state_dict["layers.0.attention.wv.weight"]
    wo = state_dict["layers.0.attention.wo.weight"]
    rms_ffn_weight = state_dict["layers.0.ffn_norm.weight"]
    w1 = state_dict["layers.0.feed_forward.w1.weight"]
    w2 = state_dict["layers.0.feed_forward.w2.weight"]
    w3 = state_dict["layers.0.feed_forward.w3.weight"]
    rms_final_weight = state_dict["norm.weight"]
    # wcls weight?

    # number of elements to extract and write to output file
    n = 4

    with open(args.fn_output, "wb") as f:
        write_array(f, tok_embeddings[0,:n].numpy())
        write_array(f, rms_att_weight[:n].numpy())
        write_array(f, wq[0,:n].numpy())
        write_array(f, wk[0,:n].numpy())
        write_array(f, wv[0,:n].numpy())
        write_array(f, wo[0,:n].numpy())
        write_array(f, rms_ffn_weight[:n].numpy())
        write_array(f, w1[0,:n].numpy())
        write_array(f, w2[0,:n].numpy())
        write_array(f, w3[0,:n].numpy())
        write_array(f, rms_final_weight[:n].numpy())

if __name__ == "__main__":
    main()

