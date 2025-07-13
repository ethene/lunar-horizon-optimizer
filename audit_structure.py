#!/usr/bin/env python3
"""
Repository Structure Audit Script

This script performs a comprehensive audit of the lunar-horizon-optimizer repository
to analyze project structure, identify active vs. outdated files, and provide
recommendations for cleanup and organization.

Uses only standard library modules for maximum compatibility.
"""

import os
import ast
import fnmatch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime


class RepositoryAuditor:
    """Comprehensive repository structure auditor."""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.report_lines = []
        self.file_analysis = {}
        self.issues_found = []
        self.recommendations = []
        
    def extract_file_header(self, file_path: Path, num_lines: int = 10) -> str:
        """Extract first N lines of a file for analysis."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= num_lines:
                        break
                    lines.append(line.rstrip())
                return '\n'.join(lines)
        except Exception as e:
            return f"Error reading file: {e}"
    
    def extract_python_docstring(self, file_path: Path) -> str:
        """Extract module docstring from Python file using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Get module docstring
            if (tree.body and 
                isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and 
                isinstance(tree.body[0].value.value, str)):
                docstring = tree.body[0].value.value.strip()
                # Return first line only for brevity
                return docstring.split('\n')[0] if docstring else "No docstring"
            
            return "No docstring"
            
        except Exception as e:
            return f"Error parsing: {e}"
    
    def analyze_python_file(self, file_path: Path) -> Dict[str, any]:
        """Analyze a Python file for structure and issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'docstring': self.extract_python_docstring(file_path),
                'line_count': len(content.split('\n')),
                'has_main': '__name__ == "__main__"' in content,
                'has_imports': any(line.strip().startswith(('import ', 'from ')) 
                                 for line in content.split('\n')),
                'has_classes': 'class ' in content,
                'has_functions': 'def ' in content,
                'issues': []
            }
            
            # Check for issues
            if 'TODO' in content:
                analysis['issues'].append('Contains TODO comments')
            if 'FIXME' in content:
                analysis['issues'].append('Contains FIXME comments')
            if 'XXX' in content:
                analysis['issues'].append('Contains XXX comments')
            if analysis['line_count'] < 5:
                analysis['issues'].append('Very short file (< 5 lines)')
            if not analysis['has_imports'] and analysis['line_count'] > 10:
                analysis['issues'].append('No imports found in substantial file')
            
            return analysis
            
        except Exception as e:
            return {
                'docstring': f"Error: {e}",
                'line_count': 0,
                'has_main': False,
                'has_imports': False,
                'has_classes': False,
                'has_functions': False,
                'issues': [f"Parse error: {e}"]
            }
    
    def get_directory_tree(self) -> Dict[str, any]:
        """Generate directory tree structure."""
        tree = {'directories': {}, 'files': []}
        
        # Get top-level items
        try:
            for item in sorted(self.repo_root.iterdir()):
                if item.name.startswith('.'):
                    continue  # Skip hidden files/dirs
                    
                if item.is_dir():
                    tree['directories'][item.name] = self.analyze_directory(item)
                else:
                    tree['files'].append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'type': self.get_file_type(item.name)
                    })
        except Exception as e:
            self.issues_found.append(f"Error reading repository root: {e}")
        
        return tree
    
    def analyze_directory(self, dir_path: Path) -> Dict[str, any]:
        """Analyze a directory and its contents."""
        analysis = {
            'files': [],
            'subdirectories': [],
            'python_files': 0,
            'test_files': 0,
            'doc_files': 0,
            'config_files': 0,
            'total_files': 0
        }
        
        try:
            for item in sorted(dir_path.iterdir()):
                if item.name.startswith('.'):
                    continue
                    
                if item.is_dir():
                    analysis['subdirectories'].append({
                        'name': item.name,
                        'file_count': len(list(item.glob('*')))
                    })
                else:
                    file_info = {
                        'name': item.name,
                        'size': item.stat().st_size,
                        'type': self.get_file_type(item.name)
                    }
                    
                    # Add content analysis for key file types
                    if item.suffix == '.py':
                        file_info['analysis'] = self.analyze_python_file(item)
                        analysis['python_files'] += 1
                        if 'test' in item.name.lower():
                            analysis['test_files'] += 1
                    elif item.suffix in ['.md', '.rst', '.txt']:
                        file_info['header'] = self.extract_file_header(item, 5)
                        analysis['doc_files'] += 1
                    elif item.suffix in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']:
                        analysis['config_files'] += 1
                    
                    analysis['files'].append(file_info)
                    analysis['total_files'] += 1
                    
        except Exception as e:
            self.issues_found.append(f"Error analyzing directory {dir_path}: {e}")
        
        return analysis
    
    def get_file_type(self, filename: str) -> str:
        """Determine file type from filename."""
        suffix = Path(filename).suffix.lower()
        
        type_mapping = {
            '.py': 'Python',
            '.md': 'Markdown',
            '.rst': 'ReStructuredText',
            '.txt': 'Text',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.cfg': 'Config',
            '.sh': 'Shell',
            '.bat': 'Batch',
            '.ps1': 'PowerShell',
            '.html': 'HTML',
            '.css': 'CSS',
            '.js': 'JavaScript',
            '.ipynb': 'Jupyter Notebook'
        }
        
        return type_mapping.get(suffix, 'Unknown')
    
    def check_mandatory_files(self) -> Dict[str, bool]:
        """Check for presence of mandatory project files."""
        mandatory_files = {
            'README.md': False,
            'requirements.txt': False,
            'environment.yml': False,
            'pyproject.toml': False,
            'setup.py': False,
            'Makefile': False,
            'CLAUDE.md': False,
            '.gitignore': False
        }
        
        for file_path in self.repo_root.iterdir():
            if file_path.name in mandatory_files:
                mandatory_files[file_path.name] = True
        
        return mandatory_files
    
    def generate_recommendations(self, tree: Dict[str, any], mandatory_files: Dict[str, bool]):
        """Generate recommendations based on analysis."""
        
        # Check for mandatory files
        missing_files = [name for name, present in mandatory_files.items() if not present]
        if missing_files:
            self.recommendations.append({
                'category': 'Missing Files',
                'items': missing_files,
                'action': 'Create these standard project files'
            })
        
        # Analyze directory structure
        structure_issues = []
        cleanup_candidates = []
        
        for dir_name, dir_info in tree['directories'].items():
            # Check for empty or nearly empty directories
            if dir_info['total_files'] == 0:
                cleanup_candidates.append(f"{dir_name}/ (empty directory)")
            elif dir_info['total_files'] == 1 and dir_info['python_files'] == 0:
                cleanup_candidates.append(f"{dir_name}/ (single non-Python file)")
            
            # Check for outdated patterns
            if 'old' in dir_name.lower() or 'backup' in dir_name.lower():
                cleanup_candidates.append(f"{dir_name}/ (appears to be backup/old)")
            
            # Check for test organization
            if dir_name == 'test' and 'tests' in tree['directories']:
                structure_issues.append("Both 'test' and 'tests' directories exist")
            
            # Check for documentation scattered across directories
            if dir_info['doc_files'] > 0 and dir_name not in ['docs', 'documentation']:
                structure_issues.append(f"Documentation files found in {dir_name}/")
        
        # Check for Python files with issues
        problem_files = []
        for dir_name, dir_info in tree['directories'].items():
            for file_info in dir_info['files']:
                if file_info['type'] == 'Python' and 'analysis' in file_info:
                    if file_info['analysis']['issues']:
                        problem_files.append(f"{dir_name}/{file_info['name']}: " + 
                                           ", ".join(file_info['analysis']['issues']))
        
        if structure_issues:
            self.recommendations.append({
                'category': 'Structure Issues',
                'items': structure_issues,
                'action': 'Review and reorganize for consistency'
            })
        
        if cleanup_candidates:
            self.recommendations.append({
                'category': 'Cleanup Candidates',
                'items': cleanup_candidates,
                'action': 'Consider archiving or removing'
            })
        
        if problem_files:
            self.recommendations.append({
                'category': 'Code Issues',
                'items': problem_files,
                'action': 'Review and fix identified issues'
            })
    
    def generate_report(self) -> str:
        """Generate the complete audit report."""
        print("🔍 Starting repository audit...")
        
        # Perform analysis
        tree = self.get_directory_tree()
        mandatory_files = self.check_mandatory_files()
        self.generate_recommendations(tree, mandatory_files)
        
        print(f"📊 Found {len(tree['directories'])} directories and {len(tree['files'])} top-level files")
        
        # Generate report
        lines = []
        lines.append("# Repository Structure Audit Report")
        lines.append("")
        lines.append("Comprehensive analysis of the lunar-horizon-optimizer repository structure.")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Repository Root**: `{self.repo_root}`")
        lines.append("")
        
        # Project Structure Section
        lines.append("## Project Structure")
        lines.append("")
        lines.append("### Top-Level Overview")
        lines.append("")
        lines.append("```")
        lines.append(f"{self.repo_root.name}/")
        
        # Top-level files
        for file_info in tree['files']:
            lines.append(f"├── {file_info['name']} ({file_info['type']}, {file_info['size']} bytes)")
        
        # Top-level directories
        for dir_name in sorted(tree['directories'].keys()):
            dir_info = tree['directories'][dir_name]
            lines.append(f"├── {dir_name}/ ({dir_info['total_files']} files, "
                        f"{len(dir_info['subdirectories'])} subdirs)")
        lines.append("```")
        lines.append("")
        
        # Mandatory Files Check
        lines.append("### Mandatory Files Status")
        lines.append("")
        lines.append("| File | Status |")
        lines.append("|------|--------|")
        for file_name, present in mandatory_files.items():
            status = "✅ Present" if present else "❌ Missing"
            lines.append(f"| `{file_name}` | {status} |")
        lines.append("")
        
        # Directory Analysis
        for dir_name, dir_info in sorted(tree['directories'].items()):
            lines.append(f"### Directory: `{dir_name}/`")
            lines.append("")
            lines.append(f"**Summary**: {dir_info['total_files']} files total "
                        f"({dir_info['python_files']} Python, {dir_info['doc_files']} docs, "
                        f"{dir_info['config_files']} config)")
            lines.append("")
            
            if dir_info['subdirectories']:
                lines.append("**Subdirectories**:")
                for subdir in dir_info['subdirectories']:
                    lines.append(f"- `{subdir['name']}/` ({subdir['file_count']} files)")
                lines.append("")
            
            if dir_info['files']:
                lines.append("**Files**:")
                lines.append("")
                
                for file_info in dir_info['files']:
                    lines.append(f"#### `{file_info['name']}` ({file_info['type']})")
                    
                    if file_info['type'] == 'Python' and 'analysis' in file_info:
                        analysis = file_info['analysis']
                        lines.append(f"**Description**: {analysis['docstring']}")
                        lines.append(f"**Lines**: {analysis['line_count']}")
                        
                        features = []
                        if analysis['has_classes']:
                            features.append("classes")
                        if analysis['has_functions']:
                            features.append("functions")
                        if analysis['has_main']:
                            features.append("main block")
                        
                        if features:
                            lines.append(f"**Contains**: {', '.join(features)}")
                        
                        if analysis['issues']:
                            lines.append(f"**Issues**: {', '.join(analysis['issues'])}")
                    
                    elif 'header' in file_info:
                        lines.append("**Content preview**:")
                        lines.append("```")
                        lines.append(file_info['header'])
                        lines.append("```")
                    
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Recommendations Section
        lines.append("## Recommendations")
        lines.append("")
        
        if self.recommendations:
            for rec in self.recommendations:
                lines.append(f"### {rec['category']}")
                lines.append("")
                lines.append(f"**Action**: {rec['action']}")
                lines.append("")
                for item in rec['items']:
                    lines.append(f"- {item}")
                lines.append("")
        else:
            lines.append("✅ No major issues found. Repository structure appears well-organized.")
            lines.append("")
        
        # Issues Section
        if self.issues_found:
            lines.append("## Issues Encountered")
            lines.append("")
            for issue in self.issues_found:
                lines.append(f"- {issue}")
            lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        total_dirs = len(tree['directories'])
        total_files = sum(d['total_files'] for d in tree['directories'].values()) + len(tree['files'])
        total_python = sum(d['python_files'] for d in tree['directories'].values())
        
        lines.append(f"- **Total directories analyzed**: {total_dirs}")
        lines.append(f"- **Total files found**: {total_files}")
        lines.append(f"- **Python files**: {total_python}")
        lines.append(f"- **Recommendations generated**: {len(self.recommendations)}")
        lines.append(f"- **Issues found**: {len(self.issues_found)}")
        lines.append("")
        
        lines.append("---")
        lines.append(f"*Report generated by audit_structure.py on {datetime.now().strftime('%Y-%m-%d')}*")
        
        return "\n".join(lines)
    
    def run_audit(self) -> str:
        """Run the complete audit and return the report."""
        report_content = self.generate_report()
        
        # Write report to file
        report_path = self.repo_root / "PROJECT_AUDIT.md"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ Audit report written to: {report_path}")
        except Exception as e:
            print(f"❌ Error writing report: {e}")
        
        return report_content


def main():
    """Main function to run the repository audit."""
    repo_root = os.getcwd()
    
    print("🔍 Repository Structure Audit")
    print("=" * 40)
    print(f"Analyzing repository: {repo_root}")
    print()
    
    auditor = RepositoryAuditor(repo_root)
    report = auditor.run_audit()
    
    print()
    print("📋 Audit Complete!")
    print("Check PROJECT_AUDIT.md for detailed results.")


if __name__ == "__main__":
    main()