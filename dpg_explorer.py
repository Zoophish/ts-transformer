import os
import sys
import numpy as np
import pandas as pd
import torch
import dearpygui.dearpygui as dpg
from datetime import datetime


base_path = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.append(base_path)
    from src.lightning.lightning_module import ProbablisticTransformerLightning
except ImportError:
    print("Warning: Could not import ProbablisticTransformerLightning.")



class TimeSeriesExplorerDPG:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.df = None
        self.ts_data = None
        self.target_col = None

        self.tags = {}
        self.plot_themes = {}

        self.plot_colors = {
            'context':      (0, 0, 255, 255),
            'ground_truth': (0, 255, 0, 255),
            'median':       (214, 40, 40, 255),
            'ci_50':        (214, 40, 40, 128),
            'ci_90':        (214, 40, 40, 77),
        }
    
    def _setup_plot_themes(self):
        """Creates DPG themes for each plot style and stores them."""
        for name, color in self.plot_colors.items():
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvAll): # apply to all item types
                    # style for lines
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                    # style for fills (used by shade_series)
                    dpg.add_theme_color(dpg.mvPlotCol_Fill, color, category=dpg.mvThemeCat_Plots)
            self.plot_themes[name] = theme

    def _log(self, message: str, level: str = 'info'):
        """Adds a color-coded message to the log console."""
        log_colors = {
            'info': (255, 255, 255, 255),
            'warning': (255, 255, 0, 255),
            'error': (255, 100, 100, 255)
        }
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        dpg.add_text(log_message, parent=self.tags['log_group'], color=log_colors.get(level, 'info'))

    def _create_ui(self):
        """Creates tags safely after context init."""
        self.tags = {
            'file_label': dpg.generate_uuid(),
            'file_rows': dpg.generate_uuid(),
            'target_col_radio': dpg.generate_uuid(),
            'target_col_group': dpg.generate_uuid(),
            'plot': dpg.generate_uuid(),
            'plot_xaxis': dpg.generate_uuid(),
            'plot_yaxis': dpg.generate_uuid(),
            'context_input': dpg.generate_uuid(),
            'horizon_input': dpg.generate_uuid(),
            'mc_samples_input': dpg.generate_uuid(),
            'start_pos_input': dpg.generate_uuid(),
            'log_window': dpg.generate_uuid(),
            'log_group': dpg.generate_uuid()
        }
        self._setup_plot_themes()
        with dpg.window(label="Main", tag="primary_window"):
            with dpg.group(horizontal=True):
                # plot and log
                with dpg.group(width=-310):
                    # plot panel
                    with dpg.child_window(label="PlotPanel", height=-150):
                        with dpg.plot(label="Timeseries Plot", height=-1, width=-1, tag=self.tags['plot']):
                            dpg.add_plot_legend()
                            self.tags['plot_xaxis'] = dpg.add_plot_axis(dpg.mvXAxis, label="Time Step")
                            self.tags['plot_yaxis'] = dpg.add_plot_axis(dpg.mvYAxis, label="Value")
                    
                    # log panel
                    with dpg.child_window(label="LogConsole", height=-1, tag=self.tags['log_window']):
                        dpg.add_group(tag=self.tags['log_group'])

                # controls panel
                with dpg.child_window(label="ControlsPanel", width=300):
                    with dpg.collapsing_header(label="Data", default_open=True):
                        dpg.add_button(label="Select File", callback=lambda: dpg.show_item("file_dialog_id"))
                        dpg.add_text("No file selected.", wrap=280, tag=self.tags['file_label'])
                        dpg.add_text("Rows = ~", wrap=280, tag=self.tags['file_rows'])
                    
                    with dpg.collapsing_header(label="Target Column", default_open=True, tag=self.tags['target_col_group']):
                        # targets added here
                        pass
                    
                    with dpg.collapsing_header(label="Inference Parameters", default_open=True):
                        dpg.add_input_int(label="Context", default_value=256, tag=self.tags['context_input'])
                        dpg.add_input_int(label="Horizon", default_value=128, tag=self.tags['horizon_input'])
                        dpg.add_input_int(label="MC Samples", default_value=16, tag=self.tags['mc_samples_input'])
                        dpg.add_input_int(label="Start Position", default_value=0, tag=self.tags['start_pos_input'])

                    dpg.add_spacer(height=10)
                    dpg.add_button(label="Recalculate & Plot", width=-1, height=40, callback=self._update_plot)
        
        # file dialogue
        with dpg.file_dialog(
            directory_selector=False, show=False, callback=self._select_file_callback,
            id="file_dialog_id", width=700, height=400
        ):
            dpg.add_file_extension("Data Files (*.csv *.parquet){.csv,.parquet}", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")

    def _clear_plot(self):
        """Clears all data series from the plot."""
        dpg.delete_item(self.tags['plot_yaxis'], children_only=True)
        dpg.set_item_label(self.tags['plot'], "Timeseries Plot")
        dpg.set_axis_limits_auto(self.tags['plot_xaxis'])
        dpg.set_axis_limits_auto(self.tags['plot_yaxis'])

    def _select_file_callback(self, sender, app_data):
        fpath = app_data['file_path_name']
        try:
            if fpath.endswith('.csv'): self.df = pd.read_csv(fpath)
            elif fpath.endswith('.parquet'): self.df = pd.read_parquet(fpath)
            else: raise ValueError("Unsupported file type")
            
            file_name = os.path.basename(fpath)
            dpg.set_value(self.tags['file_label'], file_name)
            self._update_ui_for_new_data()
            self._clear_plot()
            self._log(f"Loaded '{file_name}'. Select a target column and parameters.")
        except Exception as e:
            dpg.set_value(self.tags['file_label'], "Error loading file.")
            self._clear_plot()
            self._log(f"Failed to load file: {e}", level='error')

    def _update_ui_for_new_data(self):
        dpg.set_value(self.tags['file_rows'], f"Rows = {len(self.df)}")

        if dpg.does_item_exist(self.tags['target_col_radio']):
            dpg.delete_item(self.tags['target_col_radio'])
        
        columns = self.df.columns.tolist()
        dpg.add_radio_button(
            items=columns, tag=self.tags['target_col_radio'],
            parent=self.tags['target_col_group'],
            callback=self._set_target_col_callback,
            default_value=columns[0]
        )
        self._set_target_col_callback(None, columns[0])
        context_len = dpg.get_value(self.tags['context_input'])
        dpg.set_value(self.tags['start_pos_input'], context_len)

    def _set_target_col_callback(self, sender, app_data):
        col_name = app_data
        if self.target_col != col_name:
            self.target_col = col_name
            self.ts_data = self.df[self.target_col].to_numpy(dtype=np.float32)
            self._log(f"Target column set to: {self.target_col}")

    def _update_plot(self):
        if self.ts_data is None:
            self._log("Please select a file and a target column first.", level='warning')
            return

        context_len = dpg.get_value(self.tags['context_input'])
        horizon_len = dpg.get_value(self.tags['horizon_input'])
        start_pos = dpg.get_value(self.tags['start_pos_input'])
        mc_samples = dpg.get_value(self.tags['mc_samples_input'])
            
        if start_pos < context_len or start_pos + horizon_len > len(self.ts_data):
            self._log("Invalid Range: The chosen start position, context, and horizon are out of bounds for the data.", level='error')
            return

        self._clear_plot()
        self._log(f"Running inference for '{self.target_col}'...")

        # model inference
        X_context = torch.tensor(self.ts_data[start_pos - context_len : start_pos], dtype=torch.float32, device=self.device)
        X = X_context.repeat(mc_samples, 1).unsqueeze(-1)
        y_true = self.ts_data[start_pos : start_pos + horizon_len]
        y_pred = torch.zeros(mc_samples, horizon_len, 1)

        try:
            with torch.no_grad():
                for i in range(horizon_len):
                    out = self.model(X, is_inference=True)
                    sample = out['dist'].sample()[:, -1:, :]
                    y_pred[:, i:, :] = sample[:, :, :].cpu()
                    X = sample
                self.model.reset_kv_cache()
        except Exception as e:
            self._clear_plot()
            self._log(f"Model Inference Exception: {e}", level='error')
            return
        
        # plotting
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        q_values = np.quantile(y_pred.numpy()[:, :, 0], q=quantiles, axis=0)
        
        t_context = np.arange(start_pos - context_len, start_pos)
        t_horizon = np.arange(start_pos, start_pos + horizon_len)
        context_data = self.ts_data[start_pos - context_len : start_pos]

        series_90_ci = dpg.add_shade_series(list(t_horizon), q_values[0].tolist(), y2=q_values[4].tolist(), label='90% CI', parent=self.tags['plot_yaxis'])
        series_50_ci = dpg.add_shade_series(list(t_horizon), q_values[1].tolist(), y2=q_values[3].tolist(), label='50% CI (IQR)', parent=self.tags['plot_yaxis'])
        series_context = dpg.add_line_series(list(t_context), context_data.tolist(), label='Context', parent=self.tags['plot_yaxis'])
        series_median = dpg.add_line_series(list(t_horizon), q_values[2].tolist(), label='Median Forecast', parent=self.tags['plot_yaxis'])
        series_truth = dpg.add_line_series(list(t_horizon), y_true.tolist(), label='Ground Truth', parent=self.tags['plot_yaxis'])

        dpg.bind_item_theme(series_90_ci, self.plot_themes['ci_90'])
        dpg.bind_item_theme(series_50_ci, self.plot_themes['ci_50'])
        dpg.bind_item_theme(series_context, self.plot_themes['context'])
        dpg.bind_item_theme(series_median, self.plot_themes['median'])
        dpg.bind_item_theme(series_truth, self.plot_themes['ground_truth'])

        dpg.set_item_label(self.tags['plot'], f"Forecast for '{self.target_col}'")
        dpg.fit_axis_data(self.tags['plot_xaxis'])
        dpg.fit_axis_data(self.tags['plot_yaxis'])
        self._log("Plot updated successfully.", level='info')

    def run(self):
        dpg.create_context()
        
        self._create_ui()
        
        dpg.create_viewport(title='Explorer', width=1200, height=750)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        
        self._log("Initialised, select a file.")
        
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == '__main__':
    try:
        model_path = os.path.join(base_path, 'checkpoints', 'best.ckpt')
        if os.path.exists(model_path):
             model = ProbablisticTransformerLightning.load_from_checkpoint(model_path)
        else:
            raise RuntimeError("No checkpoint.")
    except Exception as e:
        raise RuntimeError("Could not load model.")

    model.eval()
    
    explorer = TimeSeriesExplorerDPG(model=model)
    explorer.run()
