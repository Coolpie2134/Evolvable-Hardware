"""Start the Evolvable Hardware desktop application: python app.py."""

def main():
    # Windows evaluation workers import the launch module when spawning.
    # Keep Tk and plotting imports in the parent GUI process.
    from ui.app import main as launch
    launch()


if __name__ == '__main__':
    main()
