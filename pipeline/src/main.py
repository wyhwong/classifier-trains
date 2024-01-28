import core
import env


def main():
    """
    Main function for the program
    """

    setting = core.utils.load_yml(env.CONFIG_PATH)
    facade = core.control.ModelFacade(setting=setting)
    facade.start()


if __name__ == "__main__":
    main()
