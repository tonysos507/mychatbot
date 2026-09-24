#define NOMINMAX
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

std::string read_text_from_file(const std::string& path)
{
	std::ifstream in(path, std::ios::binary);

	std::ostringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

std::vector<std::string> tokenize_words(const std::string& text)
{
	std::vector<std::string> out;
	std::string cur;
	cur.reserve(32);

	for (char ch : text)
	{
		unsigned char u = static_cast<unsigned char>(ch);
		char c = static_cast<char>(std::tolower(u));

		if (std::isalnum(static_cast<unsigned char>(c)))
		{
			cur.push_back(c);
		}
		else
		{
			if (!cur.empty())
			{
				out.push_back(cur);
				cur.clear();
			}
		}
	}

	if (!cur.empty())
	{
		out.push_back(cur);
	}

	return out;
}

struct SkipGram : torch::nn::Module
{
	torch::nn::Embedding in_embed{ nullptr };
	torch::nn::Linear out_proj{ nullptr };

	SkipGram(int64_t vocab_size, int64_t embed_dim)
	{
		in_embed = register_module("in_embed", torch::nn::Embedding(vocab_size, embed_dim));
		out_proj = register_module("out_proj", torch::nn::Linear(embed_dim, vocab_size));
	}

	torch::Tensor forward(const torch::Tensor& center_ids) // [N]
	{
		auto e = in_embed(center_ids); // [N, D]
		return out_proj(e);
	}
};

std::vector<std::string> split_chunks(const std::string& text, size_t chunk_chars = 700)
{
	std::vector<std::string> chunks;
	assert(!text.empty());

	for (size_t i = 0; i < text.size(); i += chunk_chars)
	{
		chunks.push_back(text.substr(i, std::min(chunk_chars, text.size() - i)));
	}
	return chunks;
}

torch::Tensor mean_embedding_from_tokens(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int64_t>& stoi, const torch::Tensor& embed_weight)
{
	torch::Tensor ret;

	std::vector<int64_t> ids;
	ids.reserve(tokens.size());

	for (const auto& t : tokens)
	{
		auto it = stoi.find(t);
		if (it != stoi.end())
		{
			ids.push_back(it->second);
		}
	}

	if (ids.empty())
	{
		ret = torch::zeros({embed_weight.size(1)}, torch::kFloat32);
	}
	else
	{
		auto id_tensor = torch::tensor(ids, torch::kLong);
		auto vecs = embed_weight.index_select(0, id_tensor); // [K, D]
		ret = vecs.mean(0);
	}

	return ret;
}

float cosine_sim(const torch::Tensor& a, const torch::Tensor& b)
{
	float ret = 0.0f;
	auto an = a.norm().item<float>();
	auto bn = b.norm().item<float>();

	if (an == 0.0f || bn == 0.0f)
	{
		ret = 0.0f;
	}
	else
	{
		float dot = (a * b).sum().item<float>();
		ret = dot / (an * bn);
	}
	return ret;
}

int main(int argc, char** argv)
{
	int ret = 0;

	std::string text = read_text_from_file(argv[1]);

	// 1) Build corpus tokens
	// arrays of words, like:
	// who
	// everywhere
	// program
	// program
	// ..etc
	auto corpus_tokens = tokenize_words(text);
	assert(corpus_tokens.size() > 20);
	
	// 2) Build vocab
	// one word one index, no duplication
	// 1 <--> who
	// 2 <--> everywhere
	// 3 <--> program
	// ..etc
	std::unordered_map<std::string, int64_t> stoi;
	std::vector<std::string> itos;
	stoi.reserve(corpus_tokens.size() / 2);

	for (const auto& w : corpus_tokens)
	{
		if (stoi.find(w) == stoi.end())
		{
			int64_t id = static_cast<int64_t>(itos.size());
			stoi[w] = id;
			itos.push_back(w);
		}
	}

	int64_t vocab_size = static_cast<int64_t>(itos.size());
	assert(vocab_size > 2);

	std::vector<int64_t> ids;
	ids.reserve(corpus_tokens.size());
	for (const auto& w : corpus_tokens)
	{
		ids.push_back(stoi[w]);
	}

	// 3) Create skip-gram training pairs (center -> context)
	const int window = 2;
	std::vector<int64_t> centers;
	std::vector<int64_t> contexts;
	centers.reserve(ids.size() * 2);
	contexts.reserve(ids.size() * 2);

	for (int i = 0; i < static_cast<int>(ids.size()); ++i)
	{
		for (int d = -window; d <= window; ++d)
		{
			if (d == 0)
			{
				continue;
			}

			int j = i + d;
			if (j < 0 || j >= static_cast<int>(ids.size()))
			{
				continue;
			}
			centers.push_back(ids[i]);
			contexts.push_back(ids[j]);
		}
	}

	// 4) Train Torch model
	const int64_t embed_dim = 64;
	SkipGram model(vocab_size, embed_dim);
	torch::optim::Adam opt(model.parameters(), torch::optim::AdamOptions(1e-2));

	const int batch_size = 256;
	const int epochs = 8;

	std::cout << "Training Torch embeddings... vocab=" << vocab_size << ", pairs=" << centers.size() << "\n";

	for (int epoch = 0; epoch < epochs; ++epoch)
	{
		double epoch_loss = 0.0;
		int batches = 0;

		for (size_t i = 0; i < centers.size(); i += batch_size)
		{
			size_t end = std::min(centers.size(), i + batch_size);

			std::vector<int64_t> c_batch(centers.begin() + i, centers.begin() + end);
			std::vector<int64_t> y_batch(contexts.begin() + i, contexts.begin() + end);

			auto c = torch::tensor(c_batch, torch::kLong);
			auto y = torch::tensor(y_batch, torch::kLong);

			auto logits = model.forward(c); // [B, V]
			auto loss = torch::nn::functional::cross_entropy(logits, y);

			opt.zero_grad();
			loss.backward();
			opt.step();

			epoch_loss += loss.item<double>();
			batches++;
		}

		std::cout << "Epoch " << (epoch + 1) << "/" << epochs << ", loss=" << (epoch_loss / std::max(1, batches)) << "\n";
	}

	// 5) Build chunk index using learned embeddings
	auto chunks = split_chunks(text, 700);
	std::vector<torch::Tensor> chunk_vecs;
	chunk_vecs.reserve(chunks.size());

	torch::NoGradGuard no_grad;
	auto embed_weight = model.in_embed->weight.detach().cpu(); // [V, D]

	for (const auto& ch : chunks)
	{
		auto t = tokenize_words(ch);
		chunk_vecs.push_back(mean_embedding_from_tokens(t, stoi, embed_weight));
	}

	// 6) Ask loop
	std::cout << "\nReady. Ask a question (type 'exit' to quit).\n";
	for (;;)
	{
		std::cout << "\nQ> ";
		std::string q;
		std::getline(std::cin, q);
		if (!std::cin.good() || q == "exit")
		{
			break;
		}

		auto q_tokens = tokenize_words(q);
		if (q_tokens.empty())
		{
			std::cout << "A> Please enter a valid question.\n";
			continue;
		}

		auto q_vec = mean_embedding_from_tokens(q_tokens, stoi, embed_weight);

		int best_idx = -1;
		float best_score = -2.0f;
		for (int i = 0; i < static_cast<int>(chunk_vecs.size()); ++i)
		{
			float s = cosine_sim(q_vec, chunk_vecs[i]);
			if (s > best_score)
			{
				best_score = s;
				best_idx = i;
			}
		}

		if (best_idx < 0 || best_score <= 0.0f)
		{
			std::cout << "A> I could not find a relevat answer in the file.\n";
		}
		else
		{
			std::cout << "A> " << chunks[best_idx] << "\n";
			std::cout << "(similarity=" << best_score << ")\n";
		}
	}

	return ret;
}
