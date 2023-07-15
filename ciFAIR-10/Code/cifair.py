import torchvision.datasets

class ciFAIR10(torchvision.datasets.CIFAR10):
    base_folder = 'ciFAIR-10'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip'
    filename = 'ciFAIR-10.zip'
    tgz_md5 = 'ca08fd390f0839693d3fc45c4e49585f'
    test_list = [
        ['test_batch', '01290e6b622a1977a000eff13650aca2'],
    ]
