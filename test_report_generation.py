#!/usr/bin/env python3
"""
Test script to debug report generation issues.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.report import generate_comprehensive_report

def test_report_generation():
    """Test report generation with the current evaluation data."""
    
    # Load the current evaluation data
    evaluation_path = Path("output/evaluation.json")
    
    if not evaluation_path.exists():
        print("❌ evaluation.json not found!")
        return False
    
    try:
        with open(evaluation_path, 'r') as f:
            evaluation_data = json.load(f)
        
        print("✅ Loaded evaluation data:")
        print(f"   - Summary keys: {list(evaluation_data.get('summary', {}).keys())}")
        print(f"   - Global keys: {list(evaluation_data.get('global', {}).keys())}")
        
        # Add missing fields that the report generator expects
        if 'skill_level' not in evaluation_data:
            evaluation_data['skill_level'] = 'intermediate'
            print("   - Added missing skill_level")
        
        if 'overall_score' not in evaluation_data:
            # Calculate overall score from summary
            summary = evaluation_data.get('summary', {})
            if summary:
                scores = [data.get('score', 0) for data in summary.values() if isinstance(data, dict)]
                evaluation_data['overall_score'] = sum(scores) / len(scores) if scores else 0.0
                print(f"   - Calculated overall_score: {evaluation_data['overall_score']:.1f}")
        
        # Generate reports
        print("\n🔄 Generating reports...")
        report_paths = generate_comprehensive_report(evaluation_data, "output")
        
        print("✅ Report generation completed!")
        print("   Generated files:")
        for report_type, path in report_paths.items():
            if path and Path(path).exists():
                print(f"   - {report_type.upper()}: {path}")
            else:
                print(f"   - {report_type.upper()}: FAILED")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during report generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_html_report():
    """Create a simple HTML report manually."""
    
    try:
        # Load evaluation data
        with open("output/evaluation.json", 'r') as f:
            evaluation_data = json.load(f)
        
        # Create simple HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cricket Cover Drive Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #1f77b4; color: white; padding: 20px; border-radius: 8px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .score-high {{ color: #2ca02c; font-weight: bold; }}
        .score-medium {{ color: #ff7f0e; font-weight: bold; }}
        .score-low {{ color: #d62728; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏏 Cricket Cover Drive Analysis Report</h1>
        <p>Analysis completed successfully!</p>
    </div>
    
    <div class="section">
        <h2>📊 Analysis Summary</h2>
        <p><strong>Frames Analyzed:</strong> {evaluation_data.get('global', {}).get('frames_analyzed', 0)}</p>
        <p><strong>Average FPS:</strong> {evaluation_data.get('global', {}).get('avg_fps', 0):.1f}</p>
        <p><strong>Missing Keypoints:</strong> {evaluation_data.get('global', {}).get('frames_with_missing_keypoints', 0)}</p>
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
        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
            <h3>{category}</h3>
            <p><span class="{score_class}">{score}/10</span></p>
            <p><em>{feedback}</em></p>
        </div>
"""
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>📁 Generated Files</h2>
        <ul>
            <li><strong>Annotated Video:</strong> <code>output/annotated_video.mp4</code></li>
            <li><strong>Evaluation Data:</strong> <code>output/evaluation.json</code></li>
            <li><strong>Debug Data:</strong> <code>output/debug_landmarks.csv</code></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🔧 Next Steps</h2>
        <p>To improve your cricket cover drive technique:</p>
        <ol>
            <li>Review the annotated video to see your form</li>
            <li>Focus on areas with lower scores</li>
            <li>Practice the recommended improvements</li>
            <li>Re-analyze after making changes</li>
        </ol>
    </div>
</body>
</html>
"""
        
        # Save the HTML file
        html_path = Path("output/report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Simple HTML report created: {html_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating simple HTML report: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing Report Generation...")
    print("=" * 50)
    
    # Test the main report generation
    success1 = test_report_generation()
    
    print("\n" + "=" * 50)
    
    # Create a simple HTML report as fallback
    success2 = create_simple_html_report()
    
    print("\n" + "=" * 50)
    
    if success1 or success2:
        print("✅ Report generation test completed!")
        print("📁 Check the 'output' directory for generated reports.")
    else:
        print("❌ Report generation failed!")
