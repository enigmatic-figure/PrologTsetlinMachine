from __future__ import annotations
import platform
import sys
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, DataTable
from textual.app import ComposeResult

class SystemInfoPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static('SYSTEM  Host/Runtime/Prolog/Device + live telemetry', classes='card_title')
        yield DataTable(id='system-table', zebra_stripes=True)

    def on_mount(self) -> None:
        t = self.query_one('#system-table', DataTable)
        t.add_columns('METRIC','VALUE','DETAIL')
        rows = [
            ('Host', platform.node() or 'localhost', platform.platform()),
            ('Python', platform.python_version(), sys.executable.split("\\")[-1] if "\\" in sys.executable else sys.executable.split("/")[-1]),
            ('Textual', '8.2.8', 'canonical workbench'),
            ('PTM', '0.1.0', 'scalar oracle'),
            ('CPU', '--', 'n/a'),
            ('RAM', '--', 'n/a'),
        ]
        for r in rows:
            t.add_row(*r)

    def _refresh_live(self) -> None:
        # STUB: live CPU/RAM is shown in dashboard header via app._tick_uptime; table remains static until proper DataTable cell update is implemented
        # Track as stub to avoid claiming live table data that is not yet live
        pass

    def update_env(self, env: dict) -> None:
        t = self.query_one('#system-table', DataTable)
        t.clear()
        t.add_row('Runtime', str(env.get('runtime','--')), str(env.get('backend','scalar')))
        t.add_row('GProlog', 'ok' if env.get('gprolog_available') else 'missing', str(env.get('gprolog_version','--')))
        t.add_row('Device', str(env.get('device','cpu')), str(env.get('cpu_features','--')))
        t.add_row('Workspace', str(env.get('workspace','--')), 'ptm tui')
