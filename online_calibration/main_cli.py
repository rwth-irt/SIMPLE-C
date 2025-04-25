import logging

import src.core.websocket_server
import src.local.cli_launch

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    src.core.websocket_server.main(None)  # starts ws thread
    src.local.cli_launch.main()
