#!/usr/bin/env python3
"""
Quick fix script to generate missing HTML reports.
"""

import json
from pathlib import Path

def create_html_report():
    """Create a simple HTML report from the evaluation data."""
    
    # Load evaluation data
    evaluation_path = Path("output/evaluation.json")
    
    if not evaluation_path.exists():
        print("❌ evaluation.json not found!")
        return False
    
    try:
        with open(evaluation_path, 'r') as f:
            evaluation_data = json.load(f)
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cricket Cover Drive Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #1f77b4, #ff7f0e);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: bold;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1f77b4;
            border-bottom: 3px solid #ff7f0e;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
        }}
        .score-high {{ color: #2ca02c; font-weight: bold; }}
        .score-medium {{ color: #ff7f0e; font-weight: bold; }}
        .score-low {{ color: #d62728; font-weight: bold; }}
        .success-message {{
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #c3e6cb;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏏 Cricket Cover Drive Analysis Report</h1>
        <p>Professional Biomechanical Analysis System</p>
        <p>Analysis completed successfully!</p>
    </div>
    
    <div class="success-message">
        <strong>✅ Analysis Complete!</strong> Your cricket cover drive has been successfully analyzed.
    </div>
    
    <div class="section">
        <h2>📊 Analysis Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Frames Analyzed</h3>
                <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">
                    {evaluation_data.get('global', {}).get('frames_analyzed', 0)}
                </p>
            </div>
            <div class="metric-card">
                <h3>Average FPS</h3>
                <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">
                    {evaluation_data.get('global', {}).get('avg_fps', 0):.1f}
                </p>
            </div>
            <div class="metric-card">
                <h3>Missing Keypoints</h3>
                <p style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">
                    {evaluation_data.get('global', {}).get('frames_with_missing_keypoints', 0)}
                </p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 Technique Scores</h2>
"""
        
        # Add technique scores
        summary = evaluation_data.get('summary', {})
        for category, data in summary.items():
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
                feedback = data.get('feedback', 'No feedback available')
                score_class = "score-high" if score >= 7 else "score-medium" if score >= 5 else "score-low"
                
                html_content += f"""
        <div class="metric-card" style="margin: 15px 0;">
            <h3>{category}</h3>
            <p style="font-size: 1.3em;"><span class="{score_class}">{score}/10</span></p>
            <p><em>{feedback}</em></p>
        </div>
"""
        
        # Calculate overall score
        if summary:
            scores = [data.get('score', 0) for data in summary.values() if isinstance(data, dict)]
            overall_score = sum(scores) / len(scores) if scores else 0.0
            
            # Determine skill level
            if overall_score >= 8.0:
                skill_level = "Advanced"
                skill_color = "#2ca02c"
            elif overall_score >= 6.0:
                skill_level = "Intermediate"
                skill_color = "#ff7f0e"
            else:
                skill_level = "Beginner"
                skill_color = "#d62728"
        
        html_content += f"""
    </div>
    
    <div class="section">
        <h2>🏆 Overall Assessment</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Overall Score</h3>
                <p style="font-size: 2em; font-weight: bold; color: {skill_color};">
                    {overall_score:.1f}/10
                </p>
            </div>
            <div class="metric-card">
                <h3>Skill Level</h3>
                <p style="font-size: 2em; font-weight: bold; color: {skill_color};">
                    {skill_level}
                </p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📁 Generated Files</h2>
        <ul>
            <li><strong>Annotated Video:</strong> <code>output/annotated_video.mp4</code> - Video with skeleton overlay and metrics</li>
            <li><strong>Evaluation Data:</strong> <code>output/evaluation.json</code> - Detailed analysis results</li>
            <li><strong>Debug Data:</strong> <code>output/debug_landmarks.csv</code> - Raw landmark data</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🔧 Improvement Recommendations</h2>
        <p>Based on your analysis, here are some areas to focus on:</p>
        <ol>
"""
        
        # Add specific recommendations based on scores
        for category, data in summary.items():
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
                if score < 6:
                    html_content += f"            <li><strong>{category}:</strong> Focus on improving this area (Score: {score}/10)</li>\n"
        
        html_content += """
        </ol>
        <p><strong>Next Steps:</strong></p>
        <ul>
            <li>Review the annotated video to see your form</li>
            <li>Practice the recommended improvements</li>
            <li>Record another video after making changes</li>
            <li>Compare results to track your progress</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📈 Understanding Your Scores</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>8-10: Excellent</h3>
                <p>Professional-level technique. Keep maintaining this standard.</p>
            </div>
            <div class="metric-card">
                <h3>6-7: Good</h3>
                <p>Solid technique with room for minor improvements.</p>
            </div>
            <div class="metric-card">
                <h3>4-5: Needs Work</h3>
                <p>Focus on fundamentals and practice regularly.</p>
            </div>
            <div class="metric-card">
                <h3>1-3: Beginner</h3>
                <p>Start with basic drills and seek coaching.</p>
            </div>
        </div>
    </div>
    
    <div class="section" style="text-align: center; background: #e3f2fd;">
        <h2>🎯 Keep Practicing!</h2>
        <p>Consistent practice and analysis will help you improve your cricket cover drive technique.</p>
        <p><em>Generated by Cricket Cover Drive Analysis System</em></p>
    </div>
</body>
</html>
"""
        
        # Save the HTML file
        html_path = Path("output/report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report created successfully: {html_path}")
        print(f"📊 Analysis Summary:")
        print(f"   - Frames analyzed: {evaluation_data.get('global', {}).get('frames_analyzed', 0)}")
        print(f"   - Average FPS: {evaluation_data.get('global', {}).get('avg_fps', 0):.1f}")
        print(f"   - Overall score: {overall_score:.1f}/10")
        print(f"   - Skill level: {skill_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating HTML report: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Fixing Report Generation...")
    print("=" * 50)
    
    success = create_html_report()
    
    if success:
        print("\n✅ Report generation fixed!")
        print("📁 Check 'output/report.html' for your analysis report.")
        print("🌐 Open the HTML file in your web browser to view the report.")
    else:
        print("\n❌ Failed to create report.")
