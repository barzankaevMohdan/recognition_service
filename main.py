"""
Recognition Service - Main Entry Point

Multi-camera recognition service for companies.
Automatically discovers and processes all cameras for a company.
"""

import os
import sys
import argparse
from pathlib import Path
import requests
from .multi_camera_manager import MultiCameraManager
from .logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def _load_local_env() -> None:
    """Load environment variables from recognition_service/.env if present."""
    env_path = Path(__file__).resolve().parent / '.env'
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Recognition Service - Multi-Camera Face Recognition'
    )
    
    parser.add_argument(
        '--company-slug',
        type=str,
        help='Company slug to process cameras for (or set COMPANY_SLUG)'
    )
    
    parser.add_argument(
        '--backend-url',
        type=str,
        help='Backend API URL (or set BACKEND_URL)'
    )
    
    parser.add_argument(
        '--refresh-interval',
        type=int,
        help='Interval to refresh camera list (seconds) or set REFRESH_INTERVAL'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if not args.company_slug:
        args.company_slug = os.getenv('COMPANY_SLUG')
    if not args.company_slug:
        parser.error('Missing company slug. Provide --company-slug or set COMPANY_SLUG in .env/environment.')
    
    if not args.backend_url:
        args.backend_url = os.getenv('BACKEND_URL')
    if not args.backend_url:
        parser.error('Missing backend URL. Provide --backend-url or set BACKEND_URL in .env/environment.')
    
    if args.refresh_interval is None:
        refresh_env = os.getenv('REFRESH_INTERVAL')
        if refresh_env is not None:
            try:
                args.refresh_interval = int(refresh_env)
            except ValueError:
                parser.error('REFRESH_INTERVAL must be an integer value.')
        else:
            args.refresh_interval = 60
    
    return args


def main() -> None:
    """Main entry point."""
    _load_local_env()
    args = parse_args()
    
    # Setup logging
    setup_logging('main', args.debug)
    logger = get_logger(__name__)
    
    logger.info('=' * 60)
    logger.info('Recognition Service - Multi-Camera Mode')
    logger.info('=' * 60)
    logger.info(f'Company: {args.company_slug}')
    logger.info(f'Backend: {args.backend_url}')
    logger.info(f'Refresh interval: {args.refresh_interval}s')
    logger.info('=' * 60)

    # Reset presence for this company on startup to avoid "stuck" IN statuses
    reset_token = os.getenv('PRESENCE_RESET_TOKEN')
    if not reset_token:
        logger.warning('PRESENCE_RESET_TOKEN is not set, presence reset skipped')
    else:
        try:
            reset_url = f"{args.backend_url.rstrip('/')}/api/presence/public/reset/{args.company_slug}"
            resp = requests.post(
                reset_url,
                json={"olderThanMinutes": 0},
                timeout=10,
                headers={"x-reset-token": reset_token},
            )
            resp.raise_for_status()
            logger.info(f"Presence reset for company {args.company_slug}: {resp.json()}")
        except Exception as e:
            logger.warning(f"Presence reset failed (non-fatal): {e}")
    
    try:
        manager = MultiCameraManager(
            company_slug=args.company_slug,
            backend_url=args.backend_url,
            refresh_interval=args.refresh_interval
        )
        
        manager.run()
        
    except KeyboardInterrupt:
        logger.info('Received keyboard interrupt, shutting down...')
        sys.exit(0)
    except Exception as e:
        logger.error(f'Fatal error: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
