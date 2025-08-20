"""
Report export utilities for cricket cover drive analysis.
Generates comprehensive HTML and PDF reports with analysis results.
"""

import os
import json
import base64
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def _plot_series(series: List[float], title: str, path: str):
    """Plot a time series with proper styling."""
    plt.figure(figsize=(8, 4))
    plt.plot(series, marker='o', markersize=3, linestyle='-', linewidth=1)
    plt.title(title, fontsize=12)
    plt.xlabel("Frame Number", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def build_report(output_dir: str, eval_json: Dict, timeseries: Dict[str, List[float]]):
    """Generate a comprehensive HTML report with analysis results.
    Always creates a report file, even if there's no data.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy values to Python floats in eval_json
        def convert_numpy_to_python(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(v) for v in obj]
            return obj
        
        eval_json = convert_numpy_to_python(eval_json)
        
        # Generate charts
        charts = {}
        for name, series in timeseries.items():
            if series and any([s is not None for s in series]):
                png = os.path.join(output_dir, f"{name}.png")
                _plot_series([0 if s is None else s for s in series], name, png)
                charts[name] = os.path.basename(png)

        # Generate HTML report
        html_path = os.path.join(output_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Cover Drive Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #34495e; }
                    h3 { color: #2980b9; }
                    pre { background: #f8f9fa; padding: 15px; border-radius: 5px; }
                    .chart-container { margin: 20px 0; }
                    img { max-width: 100%; height: auto; }
                    .feedback { margin: 15px 0; padding: 10px; border-radius: 5px; }
                    .good { background: #d4edda; color: #155724; }
                    .warn { background: #fff3cd; color: #856404; }
                    .bad { background: #f8d7da; color: #721c24; }
                    .score-medium { color: #ffc107; font-weight: bold; }
                    .score-low { color: #dc3545; font-weight: bold; }
                    .score-na { color: #6c757d; font-style: italic; }
                    pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
                    .chart-container { text-align: center; margin: 20px 0; }
                    img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                    .stat-card { background: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }
                    .stat-value { font-size: 24px; font-weight: bold; color: #2E86AB; }
                    .stat-label { color: #6c757d; font-size: 14px; }
                </style>
            </head>
            <body>
                <div class="container">
            """)
            
            f.write("<h1>🏏 Cricket Cover Drive Analysis Report</h1>")
            
            # Check if we have any data
            has_data = bool(eval_json) or bool(charts)
            
            if has_data:
                # Summary Scores
                f.write("<h2>📊 Summary Scores</h2>")
                summary = eval_json.get("summary", {})
                for category, data in summary.items():
                    score = data.get("score")
                    feedback = data.get("feedback", "")
                    
                    if score is None:
                        score_class = "score-na"
                        score_display = "N/A"
                    elif score >= 8:
                        score_class = "score-high"
                        score_display = f"{score}/10"
                    elif score >= 6:
                        score_class = "score-medium"
                        score_display = f"{score}/10"
                    else:
                        score_class = "score-low"
                        score_display = f"{score}/10"
                    
                    f.write(f"""
                    <div class="score-card">
                        <h3>{category}</h3>
                        <p><strong>Score:</strong> <span class="{score_class}">{score_display}</span></p>
                        <p><strong>Feedback:</strong> {feedback}</p>
                    </div>
                    """)
                
                # Global Statistics
                f.write("<h2>📈 Global Statistics</h2>")
                global_stats = eval_json.get("global", {})
                f.write('<div class="stats-grid">')
                for key, value in global_stats.items():
                    if isinstance(value, float):
                        display_value = f"{value:.2f}"
                    else:
                        display_value = str(value)
                    f.write(f"""
                    <div class="stat-card">
                        <div class="stat-value">{display_value}</div>
                        <div class="stat-label">{key.replace('_', ' ').title()}</div>
                    </div>
                    """)
                f.write('</div>')
                
                # Raw JSON data
                f.write("<h2>🔍 Raw Data</h2>")
                f.write("<pre>")
                f.write(json.dumps(eval_json, indent=2))
                f.write("</pre>")
                
                # Charts
                if charts:
                    f.write("<h2>📊 Metric Charts</h2>")
                    for name, img in charts.items():
                        f.write(f"""
                        <div class="chart-container">
                            <h3>{name.replace('_', ' ').title()}</h3>
                            <img src='{img}' alt='{name} chart'/>
                        </div>
                        """)
            else:
                # If no data, show a simple message
                f.write("""
                <div class="feedback warn">
                    <h2>No Analysis Data Available</h2>
                    <p>Please try analyzing a clearer or longer video for better results.</p>
                </div>
                """)
            
            f.write("""
                </div>
            </body>
            </html>
            """)
        
        return html_path
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        # Create a basic report even if there's an error
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <body>
            <h1>Cover Drive Analysis Report</h1>
            <p>No analysis data available.</p>
            </body>
            </html>
            """)
        return html_path


class ReportGenerator:
    """Generates comprehensive analysis reports in HTML and PDF formats."""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent.parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # Report styling
        self.css_styles = """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #1f77b4, #ff7f0e);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: bold;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #1f77b4;
            border-bottom: 3px solid #ff7f0e;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #1f77b4;
        }
        .score-high { color: #2ca02c; font-weight: bold; }
        .score-medium { color: #ff7f0e; font-weight: bold; }
        .score-low { color: #d62728; font-weight: bold; }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .recommendations {
            background: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
        }
        .recommendations h3 {
            color: #1f77b4;
            margin-top: 0;
        }
        .recommendations ul {
            margin: 10px 0;
        }
        .recommendations li {
            margin: 8px 0;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            border-top: 1px solid #ddd;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #1f77b4;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .skill-level {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }
        .skill-beginner { background-color: #d62728; }
        .skill-intermediate { background-color: #ff7f0e; }
        .skill-advanced { background-color: #2ca02c; }
        .skill-professional { background-color: #1f77b4; }
        </style>
        """
    
    def generate_html_report(self, analysis_results: Dict[str, Any], 
                         output_path: str = "output/report.html") -> str:
        """Generate comprehensive HTML report, always creating the file."""
        try:
            # Ensure directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate HTML content
            html_content = self._create_html_content(analysis_results)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            return output_path

        except Exception as e:
            print(f"[ReportGen] Error generating detailed report: {e}")

            # Fallback minimal report
            fallback_html = """
            <!DOCTYPE html>
            <html>
            <body>
            <h1>Cover Drive Analysis Report</h1>
            <p>No analysis data available.</p>
            </body>
            </html>
            """
            try:
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(fallback_html)
            except Exception as inner_e:
                print(f"[ReportGen] Failed to create fallback report: {inner_e}")

            return output_path

    
    def generate_pdf_report(self, analysis_results: Dict[str, Any], 
                          output_path: str = "output/report.pdf") -> str:
        """Generate PDF report using available PDF libraries."""
        
        if PDFKIT_AVAILABLE:
            return self._generate_pdfkit_report(analysis_results, output_path)
        elif REPORTLAB_AVAILABLE:
            return self._generate_reportlab_report(analysis_results, output_path)
        else:
            raise ImportError("No PDF generation library available. Install pdfkit or reportlab.")
    
    def _create_html_content(self, results: Dict[str, Any]) -> str:
        """Create the complete HTML content for the report."""
        
        # Extract data
        evaluation = results.get('evaluation', {})
        summary = evaluation.get('summary', {})
        global_stats = evaluation.get('global', {})
        
        # Generate HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<title>Cricket Cover Drive Analysis Report</title>",
            self.css_styles,
            "</head>",
            "<body>",
            self._create_header(evaluation),
            self._create_executive_summary(evaluation),
            self._create_technique_analysis(summary),
            self._create_detailed_metrics(results),
            self._create_performance_analysis(global_stats),
            self._create_recommendations(evaluation),
            self._create_footer(),
            "</body>",
            "</html>"
        ]
        
        return "\n".join(html_parts)
    
    def _create_header(self, evaluation: Dict[str, Any]) -> str:
        """Create the report header."""
        
        skill_level = evaluation.get('skill_level', 'Unknown')
        overall_score = evaluation.get('overall_score', 0)
        
        skill_class = f"skill-{skill_level.lower()}" if skill_level != 'Unknown' else "skill-intermediate"
        
        return f"""
        <div class="header">
            <h1>🏏 Cricket Cover Drive Analysis Report</h1>
            <p>Professional Biomechanical Analysis System</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <div style="margin-top: 20px;">
                <span class="skill-level {skill_class}">{skill_level}</span>
                <span style="margin-left: 20px; font-size: 1.2em;">
                    Overall Score: <strong>{overall_score:.1f}/10</strong>
                </span>
            </div>
        </div>
        """
    
    def _create_executive_summary(self, evaluation: Dict[str, Any]) -> str:
        """Create executive summary section."""
        
        skill_level = evaluation.get('skill_level', 'Unknown')
        overall_score = evaluation.get('overall_score', 0)
        
        summary_text = self._get_skill_level_summary(skill_level, overall_score)
        
        return f"""
        <div class="section">
            <h2>📋 Executive Summary</h2>
            <p>{summary_text}</p>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Skill Level</h3>
                    <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{skill_level}</p>
                </div>
                <div class="metric-card">
                    <h3>Overall Score</h3>
                    <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{overall_score:.1f}/10</p>
                </div>
                <div class="metric-card">
                    <h3>Analysis Date</h3>
                    <p style="font-size: 1.2em;">{datetime.now().strftime('%B %d, %Y')}</p>
                </div>
            </div>
        </div>
        """
    
    def _create_technique_analysis(self, summary: Dict[str, Any]) -> str:
        """Create technique analysis section."""
        if not summary:
            # If no summary provided, hide section entirely (instead of forcing "No data")
            return ""

        category_html = []
        for category, data in summary.items():
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
                feedback = data.get('feedback', 'No feedback available')

                score_class = (
                    "score-high" if score >= 7
                    else "score-medium" if score >= 5
                    else "score-low"
                )

                category_html.append(f"""
                <tr>
                    <td><strong>{category}</strong></td>
                    <td class="{score_class}">{score:.1f}/10</td>
                    <td>{feedback}</td>
                </tr>
                """)

        if not category_html:
            # If summary exists but no valid rows, return nothing
            return ""

        return f"""
        <div class="section">
            <h2>🎯 Technique Analysis</h2>
            <p>Detailed breakdown of technique across key categories:</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Score</th>
                        <th>Feedback</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(category_html)}
                </tbody>
            </table>
        </div>
        """

    
    def _create_detailed_metrics(self, results: Dict[str, Any]) -> str:
        """Create detailed metrics section (only if data is available)."""

        sections = []

        # Add smoothness chart if available
        if results.get('smoothness_chart') and os.path.exists(results['smoothness_chart']):
            sections.append(f"""
            <div class="chart-container">
                <h3>📊 Temporal Smoothness Analysis</h3>
                <img src="data:image/png;base64,{self._image_to_base64(results['smoothness_chart'])}" 
                    alt="Smoothness Analysis Chart">
            </div>
            """)

        # Add phase analysis if available
        phase_data = results.get('evaluation', {}).get('phase_analysis')
        if phase_data and phase_data.get('available'):
            sections.append(self._create_phase_analysis_html(phase_data))

        # Add contact analysis if available
        contact_data = results.get('evaluation', {}).get('contact_analysis')
        if contact_data and contact_data.get('available'):
            sections.append(self._create_contact_analysis_html(contact_data))

        # If no subsections, return nothing (hide section)
        if not sections:
            return ""

        return f"""
        <div class="section">
            <h2>📊 Detailed Metrics</h2>
            {''.join(sections)}
        </div>
        """

    
    def _create_phase_analysis_html(self, phase_data: Dict[str, Any]) -> str:
        """Create phase analysis HTML."""
        
        phase_scores = phase_data.get('phase_scores', {})
        
        if not phase_scores:
            return "<p>No phase analysis data available.</p>"
        
        phase_rows = []
        for phase, score in phase_scores.items():
            score_class = "score-high" if score >= 7 else "score-medium" if score >= 5 else "score-low"
            phase_rows.append(f"""
            <tr>
                <td><strong>{phase.title()}</strong></td>
                <td class="{score_class}">{score:.1f}/10</td>
            </tr>
            """)
        
        return f"""
        <h3>🏃‍♂️ Phase Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>Phase</th>
                    <th>Performance Score</th>
                </tr>
            </thead>
            <tbody>
                {''.join(phase_rows)}
            </tbody>
        </table>
        """
    
    def _create_contact_analysis_html(self, contact_data: Dict[str, Any]) -> str:
        """Create contact analysis HTML."""
        
        total_contacts = contact_data.get('total_contacts', 0)
        avg_confidence = contact_data.get('avg_confidence', 0)
        
        return f"""
        <h3>🎯 Contact Analysis</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Contacts Detected</h3>
                <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{total_contacts}</p>
            </div>
            <div class="metric-card">
                <h3>Average Confidence</h3>
                <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{avg_confidence:.2f}</p>
            </div>
        </div>
        """
    
    def _create_performance_analysis(self, global_stats: Dict[str, Any]) -> str:
        """Create performance analysis section."""
        
        if not global_stats:
            return """
            <div class="section">
                <h2>⚡ Performance Analysis</h2>
                <p>No performance data available.</p>
            </div>
            """
        
        avg_fps = global_stats.get('avg_fps', 0)
        frames_analyzed = global_stats.get('frames_analyzed', 0)
        missing_keypoints = global_stats.get('frames_with_missing_keypoints', 0)
        detection_rate = (1 - missing_keypoints / max(frames_analyzed, 1)) * 100
        
        return f"""
        <div class="section">
            <h2>⚡ Performance Analysis</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Processing Speed</h3>
                    <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{avg_fps:.1f} FPS</p>
                </div>
                <div class="metric-card">
                    <h3>Frames Analyzed</h3>
                    <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{frames_analyzed}</p>
                </div>
                <div class="metric-card">
                    <h3>Detection Rate</h3>
                    <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{detection_rate:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>Missing Keypoints</h3>
                    <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{missing_keypoints}</p>
                </div>
            </div>
        </div>
        """
    
    def _create_recommendations(self, evaluation: Dict[str, Any]) -> str:
        """Create recommendations section."""
        
        recommendations = evaluation.get('recommendations', [])
        
        if not recommendations:
            recommendations = ["No specific recommendations available."]
        
        if isinstance(recommendations, str):
            recommendations = [recommendations]
        
        rec_items = "".join([f"<li>{rec}</li>" for rec in recommendations])
        
        return f"""
        <div class="section">
            <h2>💡 Recommendations</h2>
            <div class="recommendations">
                <h3>Improvement Suggestions</h3>
                <ul>
                    {rec_items}
                </ul>
            </div>
        </div>
        """
    
    def _create_footer(self) -> str:
        """Create report footer."""
        
        return f"""
        <div class="footer">
            <p><strong>Cricket Cover Drive Analysis Report</strong></p>
            <p>Generated by AthleteRise Biomechanical Analysis System</p>
            <p>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        """
    
    def _get_skill_level_summary(self, skill_level: str, overall_score: float) -> str:
        """Get summary text based on skill level and score."""
        
        summaries = {
            'Beginner': f"This analysis indicates a {skill_level.lower()} level of technique with an overall score of {overall_score:.1f}/10. Focus on fundamental aspects of the cover drive including proper stance, head position, and basic swing mechanics.",
            'Intermediate': f"The analysis shows {skill_level.lower()} level technique with a score of {overall_score:.1f}/10. The foundation is solid with opportunities for refinement in specific areas.",
            'Advanced': f"Excellent {skill_level.lower()} level technique demonstrated with a score of {overall_score:.1f}/10. Minor adjustments can further enhance performance.",
            'Professional': f"Outstanding {skill_level.lower()} level technique with a score of {overall_score:.1f}/10. This represents elite-level execution suitable for professional competition."
        }
        
        return summaries.get(skill_level, f"Analysis completed with an overall score of {overall_score:.1f}/10.")
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for embedding in HTML."""
        
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception:
            return ""
    
    def _generate_pdfkit_report(self, results: Dict[str, Any], output_path: str) -> str:
        """Generate PDF using pdfkit."""
        
        # Create HTML first
        html_path = output_path.replace('.pdf', '_temp.html')
        self.generate_html_report(results, html_path)
        
        try:
            # Convert HTML to PDF
            pdfkit.from_file(html_path, output_path)
            
            # Clean up temporary HTML file
            os.remove(html_path)
            
            return output_path
        except Exception as e:
            raise Exception(f"PDF generation failed: {str(e)}")
    
    def _generate_reportlab_report(self, results: Dict[str, Any], output_path: str) -> str:
        """Generate PDF using reportlab."""
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Add content
        story.extend(self._create_reportlab_content(results, styles))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_reportlab_content(self, results: Dict[str, Any], styles) -> List:
        """Create content for reportlab PDF."""
        
        content = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        content.append(Paragraph("Cricket Cover Drive Analysis Report", title_style))
        content.append(Spacer(1, 20))
        
        # Add more content based on results...
        # This is a simplified version - you can expand this based on your needs
        
        return content


def create_report_generator() -> ReportGenerator:
    """Create a report generator instance."""
    return ReportGenerator()


def generate_comprehensive_report(analysis_results: Dict[str, Any], 
                                output_dir: str = "output") -> Dict[str, str]:
    """Generate comprehensive HTML and PDF reports."""
    
    generator = create_report_generator()
    output_paths = {}
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Generate HTML report
        html_path = generator.generate_html_report(
            analysis_results, 
            f"{output_dir}/report.html"
        )
        output_paths['html'] = html_path
        
        # Generate PDF report
        try:
            pdf_path = generator.generate_pdf_report(
                analysis_results, 
                f"{output_dir}/report.pdf"
            )
            output_paths['pdf'] = pdf_path
        except ImportError:
            print("Warning: PDF generation not available. Install pdfkit or reportlab.")
        except Exception as e:
            print(f"Warning: PDF generation failed: {e}")
    
    except Exception as e:
        print(f"Error generating reports: {e}")
    
    return output_paths
