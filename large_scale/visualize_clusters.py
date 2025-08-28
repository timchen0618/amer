import argparse
import numpy as np
from pathlib import Path
import json
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import pickle

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def load_clustering_results(labels_path, centroids_path, embeddings_path, retrieved_docs_path):
    """Load clustering results and embeddings."""
    # labels = np.load(labels_path)  # shape: (num_questions, num_documents)
    # centroids = np.load(centroids_path)  # shape: (num_questions, num_clusters, dimension)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    with open(centroids_path, 'rb') as f:
        centroids = pickle.load(f)
    embeddings = np.load(embeddings_path)  # shape: (num_questions, num_documents, dimension)
    print('embeddings', embeddings.shape)
    embeddings = embeddings.reshape(embeddings.shape[0], -1, embeddings.shape[-1])
    retrieved_docs = read_jsonl(retrieved_docs_path)  # list of questions with their retrieved documents
    return labels, centroids, embeddings, retrieved_docs

def create_visualization(labels, centroids, embeddings, retrieved_docs, output_path, max_docs_per_question=500):
    """Create an interactive HTML visualization of the clustering results."""
    # num_questions = len(labels)
    num_questions = 5
    
    # Prepare data for JavaScript
    plot_data = []
    questions = [doc['question'] for doc in retrieved_docs]  # Extract questions
    
    for q_idx in range(num_questions):
        # Sample documents if there are too many
        num_docs = len(embeddings[q_idx])
        print(f"Question {q_idx}: {num_docs} documents")
        if num_docs > max_docs_per_question:
            # Randomly sample documents while preserving cluster distribution
            indices = np.random.choice(num_docs, max_docs_per_question, replace=False)
            indices = np.sort(indices)  # Keep original order
            current_embeddings = embeddings[q_idx][indices]
            current_labels = labels[q_idx][indices]
            current_doc_ids = indices
        else:
            current_embeddings = embeddings[q_idx]
            current_labels = labels[q_idx]
            current_doc_ids = range(num_docs)
        
        # Project embeddings to 2D using PCA for this question
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(current_embeddings)  # shape: (num_documents, 2)
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': current_labels,
            'document_id': current_doc_ids
        })
        
        # Create traces for this question
        question_data = []
        
        # Add scatter plot for documents
        for cluster in sorted(df['cluster'].unique())[:7]:
            cluster_data = df[df['cluster'] == cluster][:20]
            question_data.append({
                'x': cluster_data['x'].tolist(),
                'y': cluster_data['y'].tolist(),
                'mode': 'markers',
                'name': f'Cluster {cluster}',
                'marker': {
                    'size': 8,
                    'color': px.colors.qualitative.Set3[cluster % len(px.colors.qualitative.Set3)]
                },
                'text': [f'Document {i}' for i in cluster_data['document_id']],
                'customdata': [{
                    'title': retrieved_docs[q_idx]['ctxs'][i]['title'] if 'title' in retrieved_docs[q_idx]['ctxs'][i] else f"Document {i}",
                    # 'title': retrieved_docs[q_idx]['ctxs'][i]['title'] if 'title' in retrieved_docs[q_idx]['ctxs'][i] else f"Document {i} | Source: {(str(retrieved_docs[q_idx]['ctxs'][i].get('source', 'unknown')).replace('_', ' ').replace('"', '').replace("'", '').replace('\\', '')[:50])}",
                    'text': retrieved_docs[q_idx]['ctxs'][i]['text'] if 'text' in retrieved_docs[q_idx]['ctxs'][i] else retrieved_docs[q_idx]['ctxs'][i]['retrieval text'].replace('_', ' ').replace('"', '').replace("'", '').replace('\\', '')
                } for i in cluster_data['document_id']],
                'hoverinfo': 'text'
            })
        
        # Add centroids
        centroids_2d = pca.transform(centroids[q_idx])  # shape: (num_clusters, 2)
        question_data.append({
            'x': centroids_2d[:, 0].tolist(),
            'y': centroids_2d[:, 1].tolist(),
            'mode': 'markers',
            'name': 'Centroids',
            'marker': {
                'size': 12,
                'symbol': 'star',
                'color': 'black'
            },
            'text': [f'Centroid {i}' for i in range(len(centroids_2d))],
            'hoverinfo': 'text'
        })
        print(question_data)
        plot_data.append(question_data)
    
    # Create the HTML with dropdown
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clustering Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            select {{ padding: 8px; font-size: 16px; margin-bottom: 20px; }}
            #documentText {{
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
                min-height: 100px;
                max-height: 300px;
                overflow-y: auto;
                white-space: pre-wrap;
                display: none;
            }}
            .document-title {{
                font-weight: bold;
                font-size: 1.1em;
                margin-bottom: 10px;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Document Clusters Visualization</h1>
            <select id="questionSelect" onchange="updateVisualization()">
                {''.join(f'<option value="{i}">Question {i + 1}</option>' for i in range(num_questions))}
            </select>
            <div id="plot"></div>
            <div id="documentText"></div>
        </div>
        
        <script>
            const plotData = {json.dumps(plot_data)};
            const questions = {json.dumps(questions)};
            let currentPlot = null;
            
            function updateVisualization() {{
                const select = document.getElementById('questionSelect');
                const questionIndex = parseInt(select.value);
                
                const layout = {{
                    title: {{
                        text: `Question ${{questionIndex + 1}}: ${{questions[questionIndex]}}`,
                        font: {{ size: 16 }},
                        y: 0.95
                    }},
                    xaxis: {{ title: 'PCA Component 1' }},
                    yaxis: {{ title: 'PCA Component 2' }},
                    hovermode: 'closest',
                    showlegend: true,
                    margin: {{ t: 100 }}  // Increase top margin to accommodate longer title
                }};
                
                if (currentPlot) {{
                    Plotly.purge('plot');
                }}
                
                currentPlot = Plotly.newPlot('plot', plotData[questionIndex], layout);
                
                // Add hover event handler
                document.getElementById('plot').on('plotly_hover', function(data) {{
                    const point = data.points[0];
                    const documentText = document.getElementById('documentText');
                    
                    if (point.customdata) {{
                        documentText.style.display = 'block';
                        documentText.innerHTML = `
                            <div class="document-title">${{point.customdata.title}}</div>
                            <div class="document-content">${{point.customdata.text}}</div>
                        `;
                    }} else {{
                        documentText.style.display = 'none';
                    }}
                }});
                
                // Hide text when not hovering
                document.getElementById('plot').on('plotly_unhover', function() {{
                    document.getElementById('documentText').style.display = 'none';
                }});
            }}
            
            // Initial plot
            updateVisualization();
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="Create visualization for clustering results")
    parser.add_argument("--labels_path", type=str, required=True,
                      help="Path to the clustering labels file")
    parser.add_argument("--centroids_path", type=str, required=True,
                      help="Path to the clustering centroids file")
    parser.add_argument("--embeddings_path", type=str, required=True,
                      help="Path to the document embeddings file")
    parser.add_argument("--retrieved_documents_path", type=str, required=True,
                      help="Path to the retrieved documents JSONL file")
    parser.add_argument("--output_path", type=str, default="clustering_visualization.html",
                      help="Path to save the visualization")
    
    args = parser.parse_args()
    
    # Load data
    labels, centroids, embeddings, retrieved_docs = load_clustering_results(
        args.labels_path,
        args.centroids_path,
        args.embeddings_path,
        args.retrieved_documents_path
    )
    
    # Create visualization
    create_visualization(labels, centroids, embeddings, retrieved_docs, args.output_path)
    print(f"Visualization saved to {args.output_path}")

if __name__ == "__main__":
    main()
    
    # python visualize_clusters.py     \
    # --labels_path /datastor1/hungting/clustering_results/mteb_retriever/stella-400M/trivia_qa_10_kmeans_10_labels.npy \
    # --centroids_path /datastor1/hungting/clustering_results/mteb_retriever/stella-400M/trivia_qa_10_kmeans_10_centroids.npy \
    # --embeddings_path /var/local/timchen0618/retrieval_outputs/echo_data/mteb_retriever/stella-400M/trivia_qa_10_doc_embeddings.npy \
    # --output_path clustering_visualization.html \
    # --retrieved_documents_path /var/local/timchen0618/retrieval_outputs/echo_data/mteb_retriever/stella-400M/trivia_qa_10.json