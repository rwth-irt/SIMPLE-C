import src.core.websocket_server
import src.local.cli_launch

if __name__ == "__main__":
    src.core.ws_sender.main()  # starts ws thread
    src.local.cli_launch.main()
