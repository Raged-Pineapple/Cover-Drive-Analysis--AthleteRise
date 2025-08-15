#!/usr/bin/env python3
"""
Test script to verify the skill grading fix.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.skill_grade import SkillGrader

def test_skill_grading():
    """Test skill grading with proper data structure."""
    
    # Create skill grader
    grader = SkillGrader()
    
    # Test with proper category scores (simple dict of scores)
    category_scores = {
        "Footwork": 7.0,
        "Head Position": 6.5,
        "Swing Control": 8.0,
        "Balance": 7.5,
        "Follow-through": 6.0
    }
    
    print("Testing skill grading with proper data structure...")
    
    try:
        assessment = grader.assess_skill_level(category_scores)
        print(f"✅ Skill grading successful!")
        print(f"   Skill Level: {assessment.skill_level.value}")
        print(f"   Overall Score: {assessment.overall_score:.1f}")
        print(f"   Strengths: {assessment.strengths}")
        print(f"   Weaknesses: {assessment.weaknesses}")
        return True
    except Exception as e:
        print(f"❌ Skill grading failed: {e}")
        return False

def test_with_summary_structure():
    """Test with the summary structure that was causing the error."""
    
    # Create skill grader
    grader = SkillGrader()
    
    # Test with summary structure (dict of dicts with 'score' and 'feedback')
    summary = {
        "Footwork": {
            "score": 7,
            "feedback": "Good footwork technique"
        },
        "Head Position": {
            "score": 6.5,
            "feedback": "Head position needs improvement"
        },
        "Swing Control": {
            "score": 8,
            "feedback": "Excellent swing control"
        },
        "Balance": {
            "score": 7.5,
            "feedback": "Good balance maintained"
        },
        "Follow-through": {
            "score": 6,
            "feedback": "Follow-through could be better"
        }
    }
    
    print("\nTesting with summary structure (should fail without fix)...")
    
    try:
        # This should fail without the fix
        assessment = grader.assess_skill_level(summary)
        print(f"❌ This should have failed but didn't!")
        return False
    except Exception as e:
        print(f"✅ Correctly failed with: {e}")
        
        # Now test the fix - extract scores
        print("\nTesting with score extraction fix...")
        category_scores = {}
        for category, data in summary.items():
            if isinstance(data, dict) and 'score' in data:
                category_scores[category] = float(data['score'])
        
        try:
            assessment = grader.assess_skill_level(category_scores)
            print(f"✅ Fix works! Skill Level: {assessment.skill_level.value}")
            return True
        except Exception as e2:
            print(f"❌ Fix failed: {e2}")
            return False

if __name__ == "__main__":
    print("Testing Skill Grading Fix")
    print("=" * 40)
    
    success1 = test_skill_grading()
    success2 = test_with_summary_structure()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The skill grading fix is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
