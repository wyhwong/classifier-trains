import pipeline.core
import pipeline.env


def main():
    """
    Main function for the program
    """

    setting = pipeline.core.utils.load_yml(pipeline.env.CONFIG_PATH)
    facade = pipeline.core.control.ModelFacade(setting=setting)
    facade.start()


if __name__ == "__main__":
    main()
