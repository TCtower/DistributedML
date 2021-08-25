
def compressor_init(args):
    if args.compression == 'dgc':
        from src.grace_dl.torch.compressor.dgc import DgcCompressor
        compressor = DgcCompressor(compress_ratio=0.01)
    elif args.compression == 'efsignsgd':
        from src.grace_dl.torch.compressor.efsignsgd import EFSignSGDCompressor
        compressor = EFSignSGDCompressor(lr=0.1)
    elif args.compression == 'fp16':
        from src.grace_dl.torch.compressor.fp16 import FP16Compressor
        compressor = FP16Compressor()
    elif args.compression == 'onebit':
        from src.grace_dl.torch.compressor.onebit import OneBitCompressor
        compressor = OneBitCompressor()
    elif args.compression == 'qsgd':
        from src.grace_dl.torch.compressor.qsgd import QSGDCompressor
        compressor = QSGDCompressor(quantum_num=64)
    elif args.compression == 'randomk':
        from src.grace_dl.torch.compressor.randomk import RandomKCompressor
        compressor = RandomKCompressor(compress_ratio=0.01)
    elif args.compression == 'signsgd':
        from src.grace_dl.torch.compressor.signsgd import SignSGDCompressor
        compressor = SignSGDCompressor()
    elif args.compression == 'signum':
        from src.grace_dl.torch.compressor.signum import SignumCompressor
        compressor = SignumCompressor(momentum=0.9)
    elif args.compression == 'terngrad':
        from src.grace_dl.torch.compressor.terngrad import TernGradCompressor
        compressor = TernGradCompressor()
    elif args.compression == 'threshold':
        from src.grace_dl.torch.compressor.threshold import ThresholdCompressor
        compressor = ThresholdCompressor(threshold=0.01)
    elif args.compression == 'topk':
        from src.grace_dl.torch.compressor.topk import TopKCompressor
        compressor = TopKCompressor(compress_ratio=args.compress_ratio)
    else:
        from src.grace_dl.torch.compressor.none import NoneCompressor
        compressor = NoneCompressor()

    return compressor
