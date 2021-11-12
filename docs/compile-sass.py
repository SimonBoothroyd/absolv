import sass


def main():
    sass.compile(dirname=("_static", "_static"), output_style="compressed")


if __name__ == "__main__":
    main()
