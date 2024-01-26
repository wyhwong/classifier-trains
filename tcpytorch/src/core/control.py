import core.training
import core.preprocessing
import core.visualization
import core.model
import core.utils


def train(args: dict[str, str]):
    data_tranforms = utils.preprocessing.get_transforms(**configs["preprocessing"])
