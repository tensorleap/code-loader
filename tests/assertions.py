from code_loader import dataset_binder
from grappa import should


def assert_dataset_binder_is_valid():
    setup_container = dataset_binder.setup_container

    len(setup_container.subsets) | should.be.higher.than(0)
    len(setup_container.inputs) | should.be.higher.than(0)
    len(setup_container.ground_truths) | should.be.higher.than(0)
    len(setup_container.metadata) | should.be.higher.than(0)
