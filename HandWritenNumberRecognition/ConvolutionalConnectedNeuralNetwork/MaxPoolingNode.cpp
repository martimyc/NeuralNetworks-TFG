#include "MaxPoolingNode.h"

#include <math.h>

#include "imgui.h"
#include "Globals.h"

MaxPoolingNode::MaxPoolingNode() :
	ComputationNode(NT_MAX_POOLING),
	window_h(sqrt(POOLING_WINDOW)),
	window_w(window_h)
{}

MaxPoolingNode::~MaxPoolingNode()
{}

void MaxPoolingNode::Forward(std::vector<Eigen::MatrixXd>& inputs) const
{
	for (std::vector<Eigen::MatrixXd>::iterator input = inputs.begin(); input != inputs.end(); input++) // Images
	{
		int cols = input->cols();
		int rows = input->rows();

		Eigen::MatrixXd mat((int)ceil((float)rows / 2.0f), (int)ceil((float)cols / 2.0f));

		for (int start_col = 0, i = 0; start_col < cols; start_col += window_h, i++)
		{
			// Define block size (h)
			int block_h;
			if (cols - start_col > window_h)
			{
				block_h = window_h;
			}
			else
			{
				block_h = cols - start_col;
			}

			for (int start_row = 0, j = 0; start_row < rows; start_row += window_w, j++)
			{
				// Define block size (w)
				int block_w;
				if(rows - start_row > window_h)
				{
					block_w = window_w;
				}
				else
				{
					block_w = rows - start_row;
				}

				// Compute Max for block
				mat(j, i) = input->block(start_row, start_col, block_w, block_h).maxCoeff();
			}
		}

		*input = mat;
	}
}

void MaxPoolingNode::Backward(const std::vector<Eigen::MatrixXd>& inputs, std::vector<Eigen::MatrixXd>& gradients)
{
	std::vector<Eigen::MatrixXd>::iterator gradient = gradients.begin();
	std::vector<Eigen::MatrixXd>::const_iterator input = inputs.begin();

	for (; gradient != gradients.end(); gradient++, input++)
	{
		Eigen::MatrixXd output (Eigen::MatrixXd::Zero(input->rows(), input->cols()));

		for (int i = 0; i < gradient->rows(); i++)
		{
			for (int j = 0; j < gradient->cols(); j++)
			{
				Eigen::MatrixXd mat(window_w, window_h);
				int mat_cols;
				int mat_rows;

				// Cols
				if ((*input).cols() - window_h * j < window_h)
				{
					mat_cols = (*input).cols() - window_h * j;
				}
				else
				{
					mat_cols = window_h;
				}

				// Rows
				if ((*input).rows() - window_w * i < window_w)
				{
					mat_rows = (*input).rows() - window_w * i;
				}
				else
				{
					mat_rows = window_w;
				}

				mat = input->block(i * window_w, j * window_h, mat_rows, mat_cols);

				// Check to what positions of mat the gradient applies to
				for (int k = 0; k < mat_cols; k++) // Cols
				{
					for (int l = 0; l < mat_rows; l++) // Rows
					{
						if(mat(l, k) == mat.maxCoeff())
						{
							output(i * window_w + l, j * window_h + k) = (*gradient)(i, j);
						}
					}
				}
			}
		}
		*gradient = output;
	}
}

bool MaxPoolingNode::UINode() const
{
	return ImGui::Button("Max\nPooling\nNode", BUTTON_SIZE);
}

void MaxPoolingNode::UIDescription() const
{
	ImGui::TextWrapped("Pooling nodes simplify the input.\nin the Max pooling case the output is the maxinmum value within a nuber of inputs.");
}
