import sys
import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import io
import scipy.stats as stats

class HeatmapPlotter:
    def __init__(self):
        self.side_length = [3, 5, 7, 9, 11]
        self.side_length_iter = 0
        self.current_cmap = "coolwarm"
        self.stamps = []
        self.ax1 = None
        self.ax2 = None
        self.square1 = None
        self.square2 = None
        self.fig = None
        self.min_value = None
        self.max_value = None
        self.mean_1 = None
        self.mean_2 = None
        self.statistics_red = {}
        self.statistics_green = {}
        self.current_rois = []
        self.peaks = False
        self.peak_boxes_visible = False
        self.peak_box_1 = None
        self.peak_box_2 = None
        self.peak_text_1 = None
        self.peak_text_2 = None
        self.p_values_red = {}
        self.p_values_green = {}

    def load_files(self, test_group, baseline, directory):
        """
        Load a group of files from the input directory.
        """
        patterns = {"file1": f"C{test_group}_Fe_yellow", "file2": f"C{test_group}_Fe_{baseline}", \
            "file3": f"C{test_group}_N_yellow", "file4": f"C{test_group}_N_{baseline}", \
            "file5": f"T{test_group}_Fe_yellow", "file6": f"T{test_group}_Fe_{baseline}", \
            "file7": f"T{test_group}_N_yellow", "file8": f"T{test_group}_N_{baseline}"}
        files = {}
        try:
            expected_keys = set(patterns.keys())
            for name in os.listdir(directory):
                for key, pattern in patterns.items():
                    if key not in files and pattern in name:
                        files[key] = name
                        if set(files.keys()) == expected_keys:
                            return files
            if len(files) < 8:
                print(f"Issue with files: Only found {len(files)} out of 8 expected files for test group {test_group}.")
                return False
            return files
        except Exception as e:
            print(f"Error loading files: {e}")
            return False

    def process_file(self, filename):
        """
        Handles initial processing of the input TSV files.
        """
        try:
            with open(filename, "r") as file:
                content = file.read().replace(",", ".").strip()
                df = pd.read_csv(StringIO(content), sep = "\t", header = None, dtype = float)
                if df.empty:
                    raise ValueError(f"File {filename} contains no data.")
                return df
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return False

    def calculate_combined_content(self, content1, content2, content3, content4):
        """
        Calculates the Fe/N ratio heatmap based on baselineing each piece of data first.
        """
        epsilon = 1e-10
        content1 = (content1 - content2.values.mean()).clip(lower = 0)
        content3 = (content3 - content4.values.mean()).clip(lower = 0).replace(0, epsilon)
        return np.log1p(content1 / content3)

    def calculate_statistics(self, combined_content, running_dict, name):
        """
        Calculates statistics for each final sample.
        """
        mean = combined_content.values.mean()
        std = combined_content.values.std()
        peak = combined_content.values.max()
        running_dict[name] = {"Mean": mean, "SD": std, "Peak": peak}

    def get_values_within_square(self, center_x, center_y, side_length, data):
        """
        Returns a list of all the pixel values within a given square.
        """
        center_x, center_y = int(center_x), int(center_y)
        vals = []
        for i in range(-int((side_length - 1 / 2)), int((side_length - 1 / 2))):
            for j in range(-int((side_length - 1 / 2)), int((side_length - 1 / 2))):
                try:
                    vals.append(data[center_x + i][center_y + j])
                except:
                    pass
        return vals

    def redraw_heatmaps(self, content_combined_1, content_combined_2):
        """
        Completely redraws the heatmaps on the display.
        """
        self.fig.clf()
        self.ax1 = self.fig.add_subplot(121)
        sns.heatmap(content_combined_1, cmap = self.current_cmap, cbar = True, cbar_kws = {"shrink": 0.5, "aspect": 10, \
        "pad": 0.02, "orientation": "vertical", "label": "Fe/N Ratio"}, ax = self.ax1, vmin = self.min_value, \
        vmax = self.max_value)
        self.ax1.set_title(f"Log1p Control Group - Average: {self.mean_1:.2f}", fontsize = 10, pad = 5)
        self.ax2 = self.fig.add_subplot(122)
        sns.heatmap(content_combined_2, cmap = self.current_cmap, cbar = True, cbar_kws = {"shrink": 0.5, "aspect": 10, \
        "pad": 0.02, "orientation": "vertical", "label": "Fe/N Ratio"}, ax = self.ax2, vmin = self.min_value, \
        vmax = self.max_value)
        self.ax2.set_title(f"Log1p TAFe-Treatment Group - Average: {self.mean_2:.2f}", fontsize = 10, pad = 5)
        self.ax1.axis("off")
        self.ax2.axis("off")
        self.ax1.set_aspect("equal", "box")
        self.ax2.set_aspect("equal", "box")
        self.square1 = patches.Rectangle((0, 0), self.side_length[self.side_length_iter], \
        self.side_length[self.side_length_iter], edgecolor = "yellow", facecolor = "none")
        self.square2 = patches.Rectangle((0, 0), self.side_length[self.side_length_iter], \
        self.side_length[self.side_length_iter], edgecolor = "yellow", facecolor = "none")
        self.ax1.add_patch(self.square1)
        self.ax2.add_patch(self.square2)
        self.square1.set_visible(False)
        self.square2.set_visible(False)
        for stamp in self.stamps:
            x, y, ax_identifier, position, text_x, text_y, roi_label, side_length = stamp
            ax = self.ax1 if ax_identifier == "ax1" else self.ax2
            new_square = patches.Rectangle(position, side_length, side_length, edgecolor = "yellow", facecolor = "none")
            ax.add_patch(new_square)
            new_text = ax.text(text_x, text_y, roi_label, color = "yellow", fontsize = 8, ha = "center", va = "bottom", \
            fontweight = "bold")
            ax.add_artist(new_text)
        self.fig.suptitle("Log1p Control Group vs. Log1p TAFe-Treatment Group", fontweight = "bold", fontsize = 20)
        self.fig.canvas.draw_idle()
        return self.fig

    def stamp_square(self, ax, square, x, y):
        """
        Places a square on the screen to highlight a region of interest.
        """
        position = square.get_xy()
        width = square.get_width()
        height = square.get_height()
        stamped_square = patches.Rectangle(position, width, height, edgecolor = "yellow", facecolor = "none")
        ax.add_patch(stamped_square)
        ax_identifier = "ax1" if ax == self.ax1 else "ax2"
        text_x = position[0] + width / 2
        text_y = position[1] + height + 3
        roi_label = f"R.O.I. {len(self.stamps) + 1}"
        ax.text(text_x, text_y, roi_label, color="yellow", fontsize = 8, ha = "center", va = "bottom", fontweight = "bold")
        self.stamps.append((x, y, ax_identifier, position, text_x, text_y, roi_label, self.side_length[self.side_length_iter]))
        self.fig.canvas.draw_idle()

    def adjust_square_size(self, event, content_combined_1, content_combined_2, iteration_direction):
        """
        Changes the stamp size to make it adjustable.
        """
        self.side_length_iter = min(max(self.side_length_iter + iteration_direction, 0), len(self.side_length) - 1)
        square, content_combined = (self.square1, content_combined_1) if event.inaxes == self.ax1 else (self.square2, \
        content_combined_2)
        vals = self.get_values_within_square(event.xdata, event.ydata, self.side_length[self.side_length_iter], \
        content_combined)
        avg, minimum, maximum = sum(vals) / len(vals), min(vals), max(vals)
        square.set_width(self.side_length[self.side_length_iter])
        square.set_height(self.side_length[self.side_length_iter])
        square.set_xy((event.xdata - self.side_length[self.side_length_iter] / 2, \
        event.ydata - self.side_length[self.side_length_iter] / 2))
        self.fig.canvas.draw_idle()

    def draw_peak_box(self, content_combined_1, content_combined_2, cmap):
        """
        Locates and draws boxes around the peaks on each display.
        """
        box_color = text_color = "yellow" if cmap == "coolwarm" else "red"
        box_size = 5
        for content_combined, ax, peak_box_attr, peak_text_attr in [(content_combined_1, self.ax1, "peak_box_1", "peak_text_1"), \
            (content_combined_2, self.ax2, "peak_box_2", "peak_text_2")]:
            peak_y, peak_x = np.unravel_index(np.argmax(content_combined.values, axis=None), content_combined.shape)
            setattr(self, peak_box_attr, patches.Rectangle((peak_x - box_size // 2, peak_y - box_size // 2), box_size, \
            box_size, edgecolor = box_color, facecolor = "none", linewidth = 2))
            ax.add_patch(getattr(self, peak_box_attr))
            setattr(self, peak_text_attr, ax.text(peak_x + 0.5, peak_y + box_size / 2 + 2, "Peak", color = text_color, \
            fontsize = 8, ha = "center", va = "top", fontweight = "bold"))
        self.peak_boxes_visible = True
        self.fig.canvas.draw_idle()

    def create_histograms(self, content_combined_1, content_combined_2, save_directory, test_group, baseline, num_bins = None):
        """
        Generates histograms for each folders contents.
        """
        max_value = max(content_combined_1.max().max(), content_combined_2.max().max())
        bin_edges = np.arange(0, math.ceil(max_value) + 1, 1) if num_bins is None else np.linspace(0, math.ceil(max_value), \
        num_bins + 1)
        hist_1, _ = np.histogram(content_combined_1, bins = bin_edges)
        hist_2, _ = np.histogram(content_combined_2, bins = bin_edges)
        hist_1, hist_2 = hist_1 / content_combined_1.size * 100, hist_2 / content_combined_2.size * 100
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharex = True, sharey = True)
        bar_width = 0.8 * np.diff(bin_edges)[0]
        for ax, hist, title in [(ax1, hist_1, "Control Group"), (ax2, hist_2, "Treatment Group")]:
            bars = ax.bar(bin_edges[:-1], hist, width = bar_width, edgecolor = "black", align = "edge")
            ax.set_title(f"Histogram for {title}", fontsize = 10, pad = 5)
            ax.set_xlabel("Pixel Value Range")
            ax.set_ylabel("Percentage of Pixels")
            ax.set_yscale("log")
            ax.set_ylim(0.01, 100)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.1, 1, 10, 100]))
            ax.yaxis.set_major_formatter(ticker.LogFormatter())
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height * 1.1, f"{height:.2f}%", ha = "center", va = "bottom", \
                    fontsize = 8, color = "black")
        plt.suptitle("Pixel Value Distributions", fontweight = "bold", fontsize = 20)
        plt.tight_layout(rect = [0, 0, 1, 0.96])
        hist_filename = os.path.join(save_directory, f"Histograms_{test_group}_{baseline}.png")
        plt.savefig(hist_filename)
        plt.close(fig)

    def create_pdf(self, save_directory, test_group, baseline):
        """
        Stores all pngs and txt files within an easier to read pdf.
        """
        pdf_filename = os.path.join(save_directory, f"Report_{test_group}_{baseline}.pdf")
        c = canvas.Canvas(pdf_filename, pagesize=letter)
        stats_filename = os.path.join(save_directory, f"Statistics_{baseline}.txt")
        with open(stats_filename, "r") as stats_file:
            text = stats_file.read()
        text_object = c.beginText(40, 750)
        text_object.setFont("Helvetica", 10)
        text_object.textLines(text)
        c.drawText(text_object)
        image_files = [f"Coolwarm_{baseline}.png", f"GreysR_{baseline}.png", f"Coolwarm_with_peaks_{baseline}.png", \
                       f"GreysR_with_peaks_{baseline}.png", f"Histograms_{test_group}_{baseline}.png"]
        max_width = 550
        for image_file in image_files:
            image_path = os.path.join(save_directory, image_file)
            if os.path.exists(image_path):
                c.showPage()
                img = Image.open(image_path)
                img_width, img_height = img.size
                max_height = max_width * (img_height / float(img_width))
                c.drawImage(image_path, 30, 500 - max_height, width = max_width, height = max_height)
        c.save()

    def end_pdf_creator(self, save_directory):
        """
        Creates a pdf of all the summary files.
        """
        pdf_filename = os.path.join(save_directory, "Final_Report.pdf")
        c = canvas.Canvas(pdf_filename, pagesize=letter)
        text_files = [os.path.join(save_directory, "P_Values_Summary.txt"), os.path.join(save_directory, "Summary_Statistics.txt")]
        for text_file in text_files:
            with open(text_file, "r") as file:
                text = file.read()
            text_object = c.beginText(40, 750)
            text_object.setFont("Helvetica", 10)
            text_object.textLines(text)
            c.drawText(text_object)
            c.showPage()
        image_files = ["Average_Mean_Values_Bar_Chart.png", "Peak_Values_Boxplot.png"]
        max_width = 550
        for image_file in image_files:
            image_path = os.path.join(save_directory, image_file)
            if os.path.exists(image_path):
                c.showPage()
                img = Image.open(image_path)
                img_width, img_height = img.size
                max_height = max_width * (img_height / float(img_width))
                c.drawImage(image_path, 30, 500 - max_height, width = max_width, height = max_height)
        c.save()

    def permutation_testing(self, content_combined_1, content_combined_2, test_group, baseline, n_permutations = 10000):
        """
        Performs permutation tests on data to find a statistical difference in means.
        """
        control, test = content_combined_1.values.flatten(), content_combined_2.values.flatten()
        combined = np.concatenate([control, test])
        observed_diff = np.mean(test) - np.mean(control)
        count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_control = combined[:len(control)]
            perm_test = combined[len(control):]
            perm_diff = np.mean(perm_test) - np.mean(perm_control)
            if perm_diff >= observed_diff:
                count += 1
        p_value = count / n_permutations
        if baseline == "red":
            self.p_values_red[test_group] = p_value
        else:
            self.p_values_green[test_group] = p_value

    def extract_stats(self, stats_dict):
        """
        Extracts statistics from the memory.
        """
        control_means = []
        test_means = []
        control_peaks = []
        test_peaks = []
        for key, stats in stats_dict.items():
            if "Control" in key:
                control_means.append(stats["Mean"])
                control_peaks.append(stats["Peak"])
            elif 'Test' in key:
                test_means.append(stats["Mean"])
                test_peaks.append(stats["Peak"])
        return control_means, test_means, control_peaks, test_peaks

    def calculate_stats(self, data):
        """
        Calculates mean and SD of data.
        """
        return np.mean(data) if data else np.nan, np.std(data) if data else np.nan

    def create_bar_chart(self, categories, averages, std_devs, save_path):
        """
        Creates a bar chart summary of the final data.
        """
        plt.figure(figsize=(8, 6))
        plt.bar(categories, averages, yerr=std_devs, capsize=5, color=["red", "red", "green", "green"])
        plt.ylabel("Average Mean Value")
        plt.title("Average Mean Values for Control and Test Groups (Red and Green Baselines)")
        plt.savefig(save_path)
        plt.close()

    def create_boxplot(self, data, categories, save_path):
        """
        Creates a boxplot summary of all the peaks.
        """
        plt.figure(figsize=(10, 7))
        plt.boxplot(data, labels = categories)
        plt.ylabel("Peak Values")
        plt.title("Distribution of Peak Values for Control and Test Groups (Red and Green Baselines)")
        plt.savefig(save_path)
        plt.close()

    def write_p_values_summary(self, p_values, file_path):
        """
        Creates a summary of all the p-values calculated during the permutation tests.
        """
        with open(file_path, "w") as p_file:
            p_file.write("Summary of P-Values\n")
            p_file.write("=" * 40 + "\n\n")
            for baseline, p_values_dict in p_values.items():
                p_file.write(f"{baseline} Baseline:\n")
                for group, p_value in p_values_dict.items():
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    p_file.write(f"Group {group}: P-Value = {p_value:.5f} ({significance})\n")
                p_file.write("\n")

    def write_summary_statistics(self, stats_dicts, file_path):
        """
        Writes the summary statistics in the final folder for all the data groups.
        """
        with open(file_path, "w") as stats_file:
            stats_file.write("Summary of All Group Statistics\n")
            stats_file.write("=" * 40 + "\n\n")
            for baseline, stats_dict in stats_dicts.items():
                stats_file.write(f"{baseline} Baseline:\n")
                for group, stats in stats_dict.items():
                    stats_file.write(f"Group {group}:\n")
                    stats_file.write(f"  Mean: {stats['Mean']:.2f}\n")
                    stats_file.write(f"  Standard Deviation: {stats['SD']:.2f}\n")
                    stats_file.write(f"  Peak: {stats['Peak']:.2f}\n\n")

    def end_file_creator(self, output_directory, test_group, baseline):
        """
        Creates all the final summary files.
        """
        save_directory = os.path.join(output_directory, "Summary_Statistics_notes")
        os.makedirs(save_directory, exist_ok = True)
        control_red_means, test_red_means, control_red_peaks, test_red_peaks = self.extract_stats(self.statistics_red)
        control_green_means, test_green_means, control_green_peaks, test_green_peaks = self.extract_stats(self.statistics_green)
        stats_red = [self.calculate_stats(control_red_means), self.calculate_stats(test_red_means)]
        stats_green = [self.calculate_stats(control_green_means), self.calculate_stats(test_green_means)]
        categories = ["Mean Control Red", "Mean Test Red", "Mean Control Green", "Mean Test Green"]
        averages = [stats[0] for stats in stats_red + stats_green]
        std_devs = [stats[1] for stats in stats_red + stats_green]
        self.create_bar_chart(categories, averages, std_devs, os.path.join(save_directory, "Average_Mean_Values_Bar_Chart.png"))
        data_peaks = [control_red_peaks, test_red_peaks, control_green_peaks, test_green_peaks]
        self.create_boxplot(data_peaks, ["Control Red", "Test Red", "Control Green", "Test Green"], \
        os.path.join(save_directory, "Peak_Values_Boxplot.png"))
        self.write_p_values_summary({"Red": self.p_values_red, "Green": self.p_values_green}, \
        os.path.join(save_directory, "P_Values_Summary.txt"))
        self.write_summary_statistics({"Red": self.statistics_red, "Green": self.statistics_green}, \
        os.path.join(save_directory, "Summary_Statistics.txt"))
        self.end_pdf_creator(save_directory)

    def save_heatmap(self, content_combined_1, content_combined_2, save_directory, filename, cmap, peaks=False):
        """
        Saves all variations of a heatmap to the groups' folders.
        """
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize = (18, 6))
        self.current_cmap = cmap
        self.redraw_heatmaps(content_combined_1, content_combined_2)
        if peaks:
            self.draw_peak_box(content_combined_1, content_combined_2, cmap=cmap)
        plt.savefig(os.path.join(save_directory, filename))
        plt.close(self.fig)

    def file_creator(self, test_group, baseline, output_directory, content_combined_1, content_combined_2):
        """
        Creates all the files within each groups files.
        """
        save_directory = os.path.join(output_directory, f"{test_group}_{baseline}_notes")
        os.makedirs(save_directory, exist_ok = True)
        heatmaps = [("Coolwarm_{baseline}.png", "coolwarm", False), ("GreysR_{baseline}.png", "Greys_r", False), \
        ("Coolwarm_with_peaks_{baseline}.png", "coolwarm", True), ("GreysR_with_peaks_{baseline}.png", "Greys_r", True)]
        for filename, cmap, peaks in heatmaps:
            self.save_heatmap(content_combined_1, content_combined_2, save_directory, filename.format(baseline = baseline), cmap, peaks)
        stats_filename = os.path.join(save_directory, f"Statistics_{baseline}.txt")
        with open(stats_filename, "w") as stats_file:
            stats_file.write(f"Statistics for Test Group {test_group} with {baseline} baseline\n")
            for group_name, content_combined in zip(["Control", "Treatment"], [content_combined_1, content_combined_2]):
                stats_file.write(f"{group_name} Group:\n")
                stats_file.write(f"  Mean: {content_combined.values.mean():.2f}\n")
                stats_file.write(f"  Std Dev: {content_combined.values.std():.2f}\n")
                stats_file.write(f"  Min: {content_combined.values.min():.2f}\n")
                stats_file.write(f"  Max: {content_combined.values.max():.2f}\n")
                stats_file.write(f"  Range: {content_combined.values.max() - content_combined.values.min():.2f}\n\n")
            stats_file.write("ROI Information:\n")
            for stamp in self.stamps:
                x, y, ax_identifier, position, text_x, text_y, roi_label, side_length = stamp
                content_combined = content_combined_1 if ax_identifier == "ax1" else content_combined_2
                vals = self.get_values_within_square(x, y, side_length, content_combined)
                avg, minimum, maximum = sum(vals) / len(vals), min(vals), max(vals)
                std_dev = np.std(vals)
                data_range = maximum - minimum
                group_name = "Control" if ax_identifier == "ax1" else "Treatment"
                stats_file.write(f"{roi_label} in {group_name} group\n")
                stats_file.write(f"  Mean: {avg:.2f}\n")
                stats_file.write(f"  Std Dev: {std_dev:.2f}\n")
                stats_file.write(f"  Min: {minimum:.2f}\n")
                stats_file.write(f"  Max: {maximum:.2f}\n")
                stats_file.write(f"  Range: {data_range:.2f}\n\n")
        self.create_histograms(content_combined_1, content_combined_2, save_directory, test_group, baseline)
        self.create_pdf(save_directory, test_group, baseline)

    def on_move(self, event, content_combined_1, content_combined_2):
        """
        Handles all the events that take place when the mouse moves.
        """
        if event.inaxes in [self.ax1, self.ax2]:
            active_square, inactive_square = (self.square1, self.square2) if event.inaxes == self.ax1 else \
            (self.square2, self.square1)
            active_square.set_visible(True)
            inactive_square.set_visible(False)
            active_square.set_xy((event.xdata - self.side_length[self.side_length_iter] / 2, \
            event.ydata - self.side_length[self.side_length_iter] / 2))
            content_combined = content_combined_1 if event.inaxes == self.ax1 else content_combined_2
            found_rois = []
            for stamp in self.stamps:
                x, y, ax_identifier, position, text_x, text_y, roi_label, side_length = stamp
                if event.inaxes == (self.ax1 if ax_identifier == "ax1" else self.ax2):
                    tolerance = 0.5
                    if position[0] - tolerance <= event.xdata <= position[0] + side_length + tolerance and \
                            position[1] - tolerance <= event.ydata <= position[1] + side_length + tolerance:
                        found_rois.append(roi_label)
                        if roi_label not in self.current_rois:
                            self.current_rois.append(roi_label)
                            vals = self.get_values_within_square(x, y, side_length, content_combined)
                            if vals:
                                avg = sum(vals) / len(vals)
                                print(f"Hovering over {roi_label} in {'Control' if ax_identifier == 'ax1' \
                                else 'Treatment'} group")
                                print(f"Mean: {avg:.2f}, SD: {np.std(vals):.2f}, Min: {min(vals):.2f}, Max: {max(vals):.2f}, Range: {max(vals) - min(vals):.2f}")
            self.current_rois = [roi for roi in self.current_rois if roi in found_rois]
        else:
            self.square1.set_visible(False)
            self.square2.set_visible(False)
            self.current_rois = []
        self.fig.canvas.draw_idle()

    def on_key(self, event, content_combined_1, content_combined_2):
        """
        Handles all the events that take place when a specific key is pressed.
        """
        if event.key in ["up", "down"]:
            direction = 1 if event.key == "up" else -1
            self.adjust_square_size(event, content_combined_1, content_combined_2, iteration_direction = direction)
        elif event.key == "enter" and event.inaxes in [self.ax1, self.ax2]:
            content_combined = content_combined_1 if event.inaxes == self.ax1 else content_combined_2
            square = self.square1 if event.inaxes == self.ax1 else self.square2
            vals = self.get_values_within_square(event.xdata, event.ydata, self.side_length[self.side_length_iter], \
            content_combined)
            if vals:
                avg = sum(vals) / len(vals)
                minimum, maximum = min(vals), max(vals)
                self.stamp_square(event.inaxes, square, event.xdata, event.ydata)
                print(f"Mean: {avg:.2f}, Min: {minimum:.2f}, Max: {maximum:.2f}")
        elif event.key == "c":
            self.current_cmap = "Greys_r" if self.current_cmap == "coolwarm" else "coolwarm"
            self.redraw_heatmaps(content_combined_1, content_combined_2)
        elif event.key == "p":
            self.draw_peak_box(content_combined_1, content_combined_2, self.current_cmap)

    def process_test_group(self, test_group, directory, output_directory):
        """
        Processes a single test group.
        """
        test_group_str = str(test_group)
        for baseline in ["green", "red"]:
            self.stamps = []
            self.current_cmap = "coolwarm"
            print(f"Processing Test Group {test_group_str} with {baseline} baseline...")
            files = self.load_files(test_group_str, baseline, directory)
            if not files or len(files) < 8:
                print(
                    f"Test Group {test_group_str} with {baseline} baseline not found or incomplete files. Skipping...")
                continue
            content = [self.process_file(os.path.join(directory, files[f"file{i}"])) for i in range(1, 9)]
            if any(df is False or df.empty for df in content):
                print(f"Test Group {test_group_str} with {baseline} baseline has errors in data files. Skipping...")
                continue
            content_combined_1 = self.calculate_combined_content(*content[:4])
            content_combined_2 = self.calculate_combined_content(*content[4:])
            stats_dict = self.statistics_red if baseline == "red" else self.statistics_green
            self.calculate_statistics(content_combined_1, stats_dict, f"Control_{test_group_str}")
            self.calculate_statistics(content_combined_2, stats_dict, f"Test_{test_group_str}")
            self.min_value, self.max_value = content_combined_1.min().min(), content_combined_2.max().max()
            self.mean_1, self.mean_2 = content_combined_1.values.mean(), content_combined_2.values.mean()
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(18, 6))
            self.redraw_heatmaps(content_combined_1, content_combined_2)
            self.fig.canvas.mpl_connect("motion_notify_event", lambda event: self.on_move(event, content_combined_1, content_combined_2))
            self.fig.canvas.mpl_connect("key_press_event", lambda event: self.on_key(event, content_combined_1, content_combined_2))
            plt.show()
            self.file_creator(test_group_str, baseline, output_directory, content_combined_1, content_combined_2)
            self.permutation_testing(content_combined_1, content_combined_2, test_group_str, baseline)
            self.end_file_creator(output_directory, test_group_str, baseline)

    def main(self):
        """
        Utilizes all previous HeatmapPlotter() methods to run the pipeline.
        """
        directory = input("What directory would you like to use as an input? ")
        output_directory = input("What directory would you like to use as an output? ")
        for test_group in range(1, 101):
            self.process_test_group(test_group, directory, output_directory)
            print(f"Finished Test Group {test_group}.")

if __name__ == "__main__":
    plotter = HeatmapPlotter()
    plotter.main()
