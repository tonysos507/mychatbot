#define NOMINMAX
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <curl/curl.h>
#include <fstream>
#include <sstream>
#include <cassert>

struct TinyGPT : torch::nn::Module
{
    torch::nn::Embedding embed{ nullptr };
    torch::nn::Linear linear{ nullptr };

    TinyGPT(int vocab_size, int embed_size)
    {
        embed = register_module("embed", torch::nn::Embedding(vocab_size, embed_size));
        linear = register_module("linear", torch::nn::Linear(embed_size, vocab_size));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = embed(x);
        x = linear(x);
        return x;
    }
};


std::string generate(TinyGPT& model, const std::string& start, std::map<char, int>& stoi, std::map<int, char>& itos, int block_size, int length)
{
    std::vector<int64_t> tokens;

    for (char c : start)
        tokens.push_back(stoi[c]);

    for (int step = 0; step < length; step++)
    {
        int begin = std::max(0, (int)tokens.size() - block_size);

        std::vector<int64_t> context(tokens.begin() + begin, tokens.end());

        auto x = torch::tensor(context, torch::kLong).unsqueeze(0);

        auto logits = model.forward(x);

        auto last_logits = logits[0][-1];

        auto probs = torch::softmax(last_logits, 0);

        auto next_token = torch::multinomial(probs, 1).item<int>();

        tokens.push_back(next_token);
    }

    std::string result;
    for (auto t : tokens)
        result += itos[t];

    return result;
}

// Read text from file
std::string read_text_from_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
//

int main(int argc, char** argv)
{
    std::string filePath = (argc > 1) ? argv[1] : "";
    std::string text = read_text_from_file(filePath);

    //    std::vector<char> chars;
    std::map<char, int> stoi;
    std::map<int, char> itos;

    for (char c : text)
    {
        if (stoi.find(c) == stoi.end())
        {
            int idx = stoi.size();
            stoi[c] = idx;
            itos[idx] = c;
        }
    }

    int vocab_size = stoi.size();
    int block_size = 8;
    int embed_size = 32;

    std::vector<int64_t> data;
    for (char c : text) data.push_back(stoi[c]);

    TinyGPT model(vocab_size, embed_size);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-2));

    for (int step = 0; step < 1000; step++)
    {
        int i = rand() % (data.size() - block_size - 1);

        std::vector<int64_t> x_data(data.begin() + i, data.begin() + i + block_size);
        std::vector<int64_t> y_data(data.begin() + i + 1, data.begin() + i + block_size + 1);

        auto x = torch::tensor(x_data, torch::kLong).unsqueeze(0);
        auto y = torch::tensor(y_data, torch::kLong).unsqueeze(0);

        auto logits = model.forward(x);                           // 1. Forward pass: input ? predictions
        auto loss = torch::nn::functional::cross_entropy(         // 2. THIS is the cost calculation
            logits.view({ -1, vocab_size }),                      // 2. Actual output (predictions)
            y.view({ -1 })                                        // 2. Expected output (ground truth)
        );

        optimizer.zero_grad();                                    // 3. Clear old gradients (reset to 0)
        loss.backward();                                          // 4. Backpropagation: compute new gradients
        optimizer.step();                                         // 5. Gradient descent: update weights

        if (step % 100 == 0)
        {
            std::cout << "loss: " << loss.item<float>() << std::endl;
        }
    }

    std::cout << generate(model, "h", stoi, itos, block_size, 50) << std::endl;

    return 0;
}